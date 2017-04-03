"""Example soft monotonic alignment decoder implementation.

This file contains an example TensorFlow implementation of the approach
described in ``Online and Linear-Time Attention by Enforcing Monotonic
Alignments''.  The function monotonic_attention covers the algorithms in the
paper and should be general-purpose.  monotonic_alignment_decoder can be used
directly in place of tf.nn.seq2seq.attention_decoder.  This implementation
attempts to deviate as little as possible from tf.nn.seq2seq.attention_decoder,
in order to facilitate comparison between the two decoders.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.util.nest import flatten
from tensorflow.python.util.nest import is_sequence


def safe_cumprod(x, **kwargs):
  """Computes cumprod in logspace using cumsum to avoid underflow."""
  return tf.exp(tf.cumsum(tf.log(tf.clip_by_value(x, 1e-10, 1)), **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
  """Compute monotonic attention distribution from choosing probabilities.

  Monotonic attention implies that the input sequence is processed in an
  explicitly left-to-right manner when generating the output sequence.  In
  addition, once an input sequence element is attended to at a given output
  timestep, elements occurring before it cannot be attended to at subsequent
  output timesteps.  This function generates attention distributions according
  to these assumptions.  For more information, see ``Online and Linear-Time
  Attention by Enforcing Monotonic Alignments''.

  Args:
    p_choose_i: Probability of choosing input sequence/memory element i.  Should
      be of shape (batch_size, input_sequence_length), and should all be in the
      range [0, 1].
    previous_attention: The attention distribution from the previous output
      timestep.  Should be of shape (batch_size, input_sequence_length).  For
      the first output timestep, preevious_attention[n] should be [1, 0, 0, ...,
      0] for all n in [0, ... batch_size - 1].
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'.
        * 'recursive' uses tf.scan to recursively compute the distribution.
          This is slowest but is exact, general, and does not suffer from
          numerical instabilities.
        * 'parallel' uses parallelized cumulative-sum and cumulative-product
          operations to compute a closed-form solution to the recurrence
          relation defining the attention distribution.  This makes it more
          efficient than 'recursive', but it requires numerical checks which
          make the distribution non-exact.  This can be a problem in particular
          when input_sequence_length is long and/or p_choose_i has entries very
          close to 0 or 1.
        * 'hard' requires that the probabilities in p_choose_i are all either 0
          or 1, and subsequently uses a more efficient and exact solution.

  Returns:
    A tensor of shape (batch_size, input_sequence_length) representing the
    attention distributions for each sequence in the batch.

  Raises:
    ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
  """
  if mode == "recursive":
    batch_size = tf.shape(p_choose_i)[0]
    # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
    shifted_1mp_choose_i = tf.concat(
        [tf.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
    # Compute attention distribution recursively as
    # q[i] = (1 - p_choose_i[i])*q[i - 1] + previous_attention[i]
    # attention[i] = p_choose_i[i]*q[i]
    attention = p_choose_i*tf.transpose(tf.scan(
        # Need to use reshape to remind TF of the shape between loop iterations
        lambda x, yz: tf.reshape(yz[0]*x + yz[1], (batch_size,)),
        # Loop variables yz[0] and yz[1]
        [tf.transpose(shifted_1mp_choose_i), tf.transpose(previous_attention)],
        # Initial value of x is just zeros
        tf.zeros((batch_size,))))
  elif mode == "parallel":
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
    # Compute recurrence relation solution
    attention = p_choose_i*cumprod_1mp_choose_i*tf.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
  elif mode == "hard":
    # Remove any probabilities before the index chosen last time step
    p_choose_i *= tf.cumsum(previous_attention, axis=1)
    # Now, use exclusive cumprod to remove probabilities after the first
    # chosen index, like so:
    # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
    # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
    # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
    attention = p_choose_i*tf.cumprod(1 - p_choose_i, axis=1, exclusive=True)
  else:
    raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
  return attention


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
    v, b, r, g = [], [], [], []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in range(num_heads):
      k = tf.get_variable("AttnW_%d" % a,
                          [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      init = tf.random_normal_initializer(stddev=1./attention_vec_size)
      v.append(tf.get_variable(
          "AttnV_%d" % a, [attention_vec_size], initializer=init))
      r.append(tf.get_variable(
          "AttnR_%d" % a, [],
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

    def attention(query, previous_attentions):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      alignments = []
      if is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = tf.concat(1, query_list)
      for a in range(num_heads):
        with tf.variable_scope("Attention_%d" % a) as scope:
          previous_attention = previous_attentions[a]
          y = tf.contrib.layers.linear(query, attention_vec_size, scope=scope,
                                       biases_initializer=None)
          y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
          normed_v = g[a]*v[a]/tf.norm(v[a])
          # Attention mask is a softmax of v^T * tanh(...).
          s = tf.reduce_sum(normed_v*tf.tanh(hidden_features[a] + y + b[a]),
                            [2, 3])
          s += r[a]
          if hard_sigmoid:
            # At test time (i.e. not computing gradients), use hard sigmoid
            a = tf.cast(tf.greater(s, 0.), s.dtype)
            attention = monotonic_attention(a, previous_attention, "hard")
          else:
            a = tf.nn.sigmoid(
                s + sigmoid_noise_std_dev*tf.random_normal(tf.shape(s)))
            attention = monotonic_attention(a, previous_attention, "recursive")
          alignments.append(attention)
          # Now calculate the attention-weighted vector d.
          d = tf.reduce_sum(
              tf.reshape(attention, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(tf.reshape(d, [-1, attn_size]))
      return ds, alignments

    outputs = []
    prev = None
    batch_attn_size = tf.stack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in range(num_heads)]
    # Initialize the first alignment to dirac distributions which will cause
    # the attention to compute the right thing without special casing
    all_alignments = [
        [tf.one_hot(tf.zeros((batch_size,), tf.int32), attn_length, dtype=dtype)
         for _ in range(num_heads)]]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns, alignments = attention(initial_state, all_alignments[-1])
      all_alignments.append(alignments)
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
      input_size = input_size.value
      x = tf.contrib.layers.linear(tf.concat([inp] + attns, 1), input_size,
                                   reuse=i > 0, scope=tf.get_variable_scope())
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          attns, alignments = attention(state, all_alignments[-1])
      else:
        attns, alignments = attention(state, all_alignments[-1])
      all_alignments.append(alignments)

      with tf.variable_scope("AttnOutputProjection"):
        output = tf.contrib.layers.linear(
            tf.concat([cell_output] + attns, 1), output_size, reuse=i > 0,
            scope=tf.get_variable_scope())
      if loop_function is not None:
        prev = output
      outputs.append(output)
  return outputs, state
