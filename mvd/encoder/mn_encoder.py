from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder import *
from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'MNEncoder',
]

class MNEncoder(Encoder):
  """Memory Network Encoder
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
    #[batch_size, max_qa_len, max_q_len, word_dim]
    history = input_data['h_qa']
    #[batch_size, max_qa_len]
    history_len = input_data['h_qa_len']
    batch_size = tf.shape(history)[0]
    max_qa_len = tf.shape(history)[1]
    max_q_len = tf.shape(history)[2]
    word_dim = int(history.get_shape()[3])
    #reshape
    reshape_history = tf.reshape(history, [-1, max_q_len, word_dim])
    reshape_history_len = tf.reshape(history_len, [-1])
    cells = self.rnn_cells(params)

    if params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, state = tf.nn.dynamic_rnn(cell, reshape_history,
                                          sequence_length=reshape_history_len, dtype=tf.float32, scope='history_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      output_feature = tf.concat(output_states, axis=-1)
    else:
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], reshape_history, 
                                      sequence_length=reshape_history_len, dtype=tf.float32, scope='history_bidirectional_rnn')
      output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)
    history_dim = int(output_feature.get_shape()[-1])
    #[batch_size, max_qa_len, history_dim]
    output_feature = tf.reshape(output_feature, [batch_size, max_qa_len, history_dim])
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

  def memory_attention(self, query, memory, memory_length):
    """
    Args:
      query: [batch_size, out_dim]
      memory: [batch_size, max_mem_len, out_dim]
      memory_length: [batch_size]
    Returns:
      out_feature: [batch_size, out_dim]
    """
    out_dim = int(query.get_shape()[1])
    max_mem_len = tf.shape(memory)[1]
    score = tf.reduce_sum(query[:, None, :] * memory, 2)
    mask = tf.sequence_mask(memory_length, max_mem_len)
    score_mask_values = float('-inf') * tf.ones_like(score)
    new_score = tf.where(mask, score, score_mask_values)
    #[batch_size, max_mem_len]
    align = tf.nn.softmax(new_score)
    #[batch_size, out_dim]
    atten_memory = tf.reduce_sum(align[:, :, None] * memory, 1)

    reshape_atten_memory = tf.layers.dense(atten_memory, out_dim, name='atten_memory_reshape')

    return reshape_atten_memory + query


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
    #encode question
    with tf.variable_scope('question') as scope:
      question_feature = self.question_encode(input_data, self.params['question_encoder_params'])
    features.append(question_feature)
    #[batch_size, qv_dims]
    qv_feature = tf.concat(features, axis=1)
    #rehshape dims
    encode_out_dim = self.params['encode_out_dim']
    #[batch_size, out_dim]
    reshape_qv_feature = tf.layers.dense(qv_feature, encode_out_dim, name='qv_reshape')
    #encode history
    with tf.variable_scope('history') as scope:
      #[batch_size, max_qa_len, history_dim]
      history_feature = self.history_encode(input_data, self.params['history_encoder_params'])
    #memory=-network
    with tf.variable_scope('memory_attention') as scope:
      memory_length = input_data['h_len']
      encode_output = self.memory_attention(reshape_qv_feature, history_feature, memory_length)
    return encode_output
