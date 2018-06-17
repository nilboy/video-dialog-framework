from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.decoder import *

import tensorflow as tf

__all__ = [
  'DDecoder',
]

class DDecoder(Decoder):
  """Discriminative Decoder
  """

  def rnn_cells(self):
    cell_class = eval(self.params['cell'])
    cells = []
    for i in range(self.params['layer_num']):
      cell = cell_class(self.params['num_units'], reuse=tf.get_variable_scope().reuse)
      if self.params['is_train']:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.params['dropout_keep_rate'])
      cells.append(cell)
    return cells


  def decode(self, input_data, encode_output):
    """
    Args:
      input_data: dict of tensor
      encode_output: [batch_size, encode_out_dims]
    Returns:
      logits: [batch_size, candidate_num]
    """
    #[batch_size, candidate_num, max_a_len, word_dim]
    candidate_a = input_data['candidate_a']
    #[batch_size, candidate_num]
    candidate_a_len = input_data['candidate_a_len']
    #shape
    candidate_num = int(candidate_a.get_shape()[1])
    word_dim = int(candidate_a.get_shape()[3])
    batch_size = tf.shape(candidate_a)[0]
    max_a_len = tf.shape(candidate_a)[2]
    #[batch_size * candidate_num, max_a_len, word_dim]
    reshape_candidate_a = tf.reshape(candidate_a, [-1, max_a_len, word_dim])
    reshape_candidate_a_len = tf.reshape(candidate_a_len, [-1])
    #construct rnn
    cells = self.rnn_cells()
    if self.params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, state = tf.nn.dynamic_rnn(cell, reshape_candidate_a,
                                          sequence_length=reshape_candidate_a_len, dtype=tf.float32, scope='dynamic_rnn')
      output_states = []
      for i in range(self.params['layer_num']):
        output_states.append(state[i].h)
      output_feature = tf.concat(output_states, axis=-1)
    else:
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], reshape_candidate_a, 
                                      sequence_length=reshape_candidate_a_len, dtype=tf.float32, scope='bidirectional_rnn')
      output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)
    #
    output_dim = int(output_feature.get_shape()[-1])
    #reshape [batch_size, candidate_num, output_dim]
    reshape_output_feature = tf.reshape(output_feature, [batch_size, candidate_num, output_dim])
    #[batch_size, 1, encode_out_dims]
    encode_output = encode_output[:, None, :]
    encode_out_dims = int(encode_output.get_shape()[-1])
    #assert output_dim == encode_out_dims, 'encode_out_dims != decode_out_dims'
    """
    if output_dim != encode_out_dims:
      encode_output = tf.layers.dense(encode_output, output_dim, name='decoder_reshape')
    """
    reshape_output_feature = tf.layers.dense(reshape_output_feature, encode_out_dims, name='decoder_reshape')
    #[batch_size, candidate_num]
    logits = tf.reduce_sum(reshape_output_feature * encode_output, axis=2)
    return logits





