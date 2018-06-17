from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder import *
from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'LFEncoder',
]

class LFEncoder(Encoder):
  """Late Fusion Encoder
  """

  def rnn_cells(self, params):
    cell_class = eval(params['cell'])
    cells = []
    for i in range(params['layer_num']):
      cell = cell_class(params['num_units'], reuse=tf.get_variable_scope().reuse)
      if self.params['is_train']:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=params['dropout_keep_rate'])
      cells.append(cell)
    return cells

  def video_encode(self, input_data, params):
    video_feature_list = []
    if self.params['vgg']:
      vgg_encoder = construct_video_encoder(params, 'vgg')
      vgg_feature = vgg_encoder(input_data['vgg'])
      video_feature_list.append(vgg_feature)
    if self.params['c3d']:
      c3d_encoder = construct_video_encoder(params, 'c3d')
      c3d_feature = c3d_encoder(input_data['c3d'])
      video_feature_list.append(c3d_feature)
    #[batch_size, video_feature_dim]
    video_feature = tf.concat(video_feature_list, axis=1)
    return video_feature

  def history_encode(self, input_data, params):
    #encode_history
    #[batch_size, max_len, word_dim]
    history = input_data['h_all']
    history_len = input_data['h_all_len']
    cells = self.rnn_cells(params)

    if params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, state = tf.nn.dynamic_rnn(cell, history,
                                          sequence_length=history_len, dtype=tf.float32, scope='history_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      output_feature = tf.concat(output_states, axis=-1)
    else:
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], history, 
                                      sequence_length=history_len, dtype=tf.float32, scope='history_bidirectional_rnn')
      output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)

    return output_feature


  def question_encode(self, input_data, params):
    #encode_question
    #[batch_size, max_q_len, word_dim]
    question = input_data['q']
    question_len = input_data['q_len']
    cells = self.rnn_cells(params)

    if params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, state = tf.nn.dynamic_rnn(cell, question,
                                          sequence_length=question_len, dtype=tf.float32, scope='question_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      output_feature = tf.concat(output_states, axis=-1)
    else:
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], question, 
                                      sequence_length=question_len, dtype=tf.float32, scope='question_bidirectional_rnn')
      output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)

    return output_feature

  def encode(self, input_data):
    """
    Args:
      input_data: dict of tensor
    Returns:
      encode_output: [batch_size, out_dim]
    """
    features = []
    #encode video
    if self.params['video']:
      self.params['video_encoder_params']['is_train'] = self.params['is_train']
      with tf.variable_scope('video') as scope:
        video_feature = self.video_encode(input_data, self.params['video_encoder_params'])
      features.append(video_feature)
    #encode history
    if self.params['history']:
      with tf.variable_scope('history') as scope:
        history_feature = self.history_encode(input_data, self.params['history_encoder_params'])
      features.append(history_feature)
    #encode question
    with tf.variable_scope('question') as scope:
      question_feature = self.question_encode(input_data, self.params['question_encoder_params'])
    features.append(question_feature)
    merge_feature = tf.concat(features, axis=1)
    #reshape dims
    encode_out_dim = self.params['encode_out_dim']
    encode_output = tf.layers.dense(merge_feature, encode_out_dim, name='encode_reshape')
    return encode_output