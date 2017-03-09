"""Example soft monotonic alignment decoder implementation.

Can be used directly in place of tf.nn.seq2seq.attention_decoder.  This
implementation attempts to deviate as little as possible from
tf.nn.seq2seq.attention_decoder, in order to facilitate comparison between the
two decoders. """

import tensorflow as tf
from tensorflow.python.util.nest import is_sequence, flatten
import numpy as np


def safe_cumprod(x, **kwargs):
  """Computes cumprod in logspace using cumsum to avoid underflow."""
  return tf.exp(tf.cumsum(tf.log(tf.clip_by_value(x, 1e-10, 1)), **kwargs))


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Note: Copied from
  https://github.com/tensorflow/tensorflow/blob/c2c4c208679305d6d538255be569a2822f1c920f/tensorflow/python/ops/rnn_cell_impl.py#L821

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = tf.get_variable_scope()
  with tf.variable_scope(scope) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)


def monotonic_alignment_decoder(
    decoder_inputs, initial_state, attention_states, cell, output_size=None,
    num_heads=1, loop_function=None, dtype=None, scope=None,
    initial_state_attention=False, sigmoid_noise_std_dev=0.,
    initial_energy_bias=0., initial_energy_gain=None, hard_sigmoid=False):
  """RNN decoder with monotonic alignment for the sequence-to-sequence model.

  In this context "monotonic alignment" means that, during decoding, the RNN can
  look up information in the additional tensor attention_states, and it does
  this by focusing on a few entries from the tensor.  The attention mechanism
  used here is such that the first element in attention_states which has a high
  coefficient is likely to be chosen, and the subsequent attentions will only
  look at items from attention_state after the one chosen at a previous step.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.
    sigmoid_noise_std_dev: Standard deviation of pre-sigmoid additive noise.  To
      ensure that the model produces hard alignments, this should be set larger
      than 0.
    initial_energy_bias: Initial value for bias scalar in energy computation.
      Setting this value negative (e.g. -4) ensures that the initial attention
      is spread out across the encoder states at the beginning of training,
      which can facilitate convergence.
    initial_energy_gain: Initial gain term scalar in energy computation.
      Setting this value too large may result in the attention sigmoids becoming
      saturated and losing the learning signal.  By default, it is set to
      1/sqrt(attn_size).
    hard_sigmoid: Whether to use a hard sigmoid when computing attention
      probabilities.  This should be set to False during training, and True
      during testing to simulate linear time/online computation.
  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with tf.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = tf.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v, b, c, g = [], [], [], []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in range(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                          [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      init = tf.random_normal_initializer(stddev=1./attention_vec_size)
      v.append(tf.get_variable(
          "AttnV_%d" % a, [attention_vec_size], initializer=init))
      c.append(tf.get_variable(
          "AttnC_%d" % a, [],
          initializer=tf.constant_initializer(initial_energy_bias)))
      b.append(tf.get_variable(
          "AttnB_%d" % a, [attention_vec_size],
          initializer=tf.zeros_initializer()))
      if initial_energy_gain is None:
        initial_energy_gain = np.sqrt(1./attention_vec_size)
      g.append(tf.get_variable(
          "AttnG_%d" % a, [],
          initializer=tf.constant_initializer(initial_energy_gain)))

    state = initial_state

    def attention(query, masks):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      output_masks = []
      if is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = tf.concat(1, query_list)
      for a in range(num_heads):
        with tf.variable_scope("Attention_%d" % a):
          mask = masks[a]
          y = linear(query, attention_vec_size, True)
          y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
          normed_v = g[a]*v[a]/tf.sqrt(tf.reduce_sum(v[a]**2))
          # Attention mask is a softmax of v^T * tanh(...).
          s = tf.reduce_sum(normed_v*tf.tanh(hidden_features[a] + y + b[a]),
                            [2, 3])
          s += c[a]
          if hard_sigmoid:
            a = tf.cast(tf.greater(s, 0.), s.dtype)
          else:
            a = tf.nn.sigmoid(
                s + sigmoid_noise_std_dev*tf.random_normal(tf.shape(s)))
          a *= safe_cumprod(1 - a, axis=1, exclusive=True)
          a *= mask
          mask = tf.cumsum(a, axis=1, exclusive=False)
          output_masks.append(mask)
          # Now calculate the attention-weighted vector d.
          d = tf.reduce_sum(
              tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(tf.reshape(d, [-1, attn_size]))
      return ds, output_masks

    outputs = []
    prev = None
    batch_attn_size = tf.stack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in range(num_heads)]
    # This soft mask prevents the mechanism from choosing the same entry again
    masks = [tf.ones((batch_size, attn_length), dtype=dtype)
             for _ in range(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns, masks = attention(initial_state, masks)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          attns, masks = attention(state, masks)
      else:
        attns, masks = attention(state, masks)

      with tf.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)
  return outputs, state
