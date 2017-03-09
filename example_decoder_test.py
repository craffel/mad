"""Simple test for example_decoder.py"""

import tensorflow as tf
from example_decoder import monotonic_alignment_decoder


def test_monotonic_alignment_decoder():
  """Test for utils.learning_to_emit_decoder."""
  with tf.Session() as sess:
    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
      cell = tf.contrib.rnn.GRUCell(2)
      inp = [tf.constant(0.5, shape=[2, 2])] * 2
      enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
          cell, inp, dtype=tf.float32)
      attn_states = tf.concat(
          [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)
      dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
      dec, mem = monotonic_alignment_decoder(
          dec_inp, enc_state,
          attn_states, cell, output_size=4)
      sess.run([tf.global_variables_initializer()])
      res = sess.run(dec)
      assert len(res) == 3
      assert res[0].shape == (2, 4)

      res = sess.run([mem])
      assert res[0].shape == (2, 2)


if __name__ == '__main__':
  test_monotonic_alignment_decoder()
