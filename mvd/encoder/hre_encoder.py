from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder import *
from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'HREEncoder',
]

class HREEncoder(Encoder):
  """Hierarchical Recurrent Encoder
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

  def qs_q_encode(self, questions, questions_len, question, question_len, params):
    """
    Args:
      questions: [batch_size, max_qa_len, max_q_len, dims]
      questions_len: [batch_size, max_qa_len]
      question: [batch_size, max_q_len, dims]
      question_len: [batch_size]
    Returns:
      questions_output_feature: [batch_size, max_qa_len, question_dim]
      question_output_feature: [batch_size, question_dim]
    """
    batch_size = tf.shape(questions)[0]
    max_qa_len = tf.shape(questions)[1]
    max_q_len = tf.shape(questions)[2]
    dims = int(questions.get_shape()[-1])
    #
    reshape_questions = tf.reshape(questions, [-1, max_q_len, dims])
    reshape_questions_len = tf.reshape(questions_len, [-1])
    #encode_history
    cells = self.rnn_cells(params)

    if params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      #questions
      outputs, state = tf.nn.dynamic_rnn(cell, reshape_questions,
                                          sequence_length=reshape_questions_len, dtype=tf.float32, scope='questions_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      questions_output_feature = tf.concat(output_states, axis=-1)
      #question
      outputs, state = tf.nn.dynamic_rnn(cell, question,
                                          sequence_length=question_len, dtype=tf.float32, scope='question_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      question_output_feature = tf.concat(output_states, axis=-1)

    else:
      #questions
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], reshape_questions, 
                                      sequence_length=reshape_questions_len, dtype=tf.float32, scope='questions_bidirectional_rnn')
      questions_output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)
      #question
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], question, 
                                      sequence_length=question_len, dtype=tf.float32, scope='question_bidirectional_rnn')
      question_output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)

    output_dim = int(questions_output_feature.get_shape()[-1])
    #[batch_size, max_qa_len, question_dim]
    questions_output_feature = tf.reshape(questions_output_feature, [batch_size, max_qa_len, output_dim])
    return questions_output_feature, question_output_feature

  def dialog_encode(self, dialog, dialog_len, params):
    """
    Args:
      dialog: [batch_size, max_qa_len, merge_dim]
      dialog_len [batch_size]
    Returns:
      output_feature: [batch_size, out_dim]
    """
    #encode_history
    cells = self.rnn_cells(params)

    if params['stack_type'] == 'stack':
      cell = tf.contrib.rnn.MultiRNNCell(cells)
      outputs, state = tf.nn.dynamic_rnn(cell, dialog,
                                          sequence_length=dialog_len, dtype=tf.float32, scope='dialog_dynamic_rnn')
      output_states = []
      for i in range(params['layer_num']):
        output_states.append(state[i].h)
      output_feature = tf.concat(output_states, axis=-1)
    else:
      outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], dialog, 
                                      sequence_length=dialog_len, dtype=tf.float32, scope='dialog_bidirectional_rnn')
      output_feature = tf.concat([output_states[0].h, output_states[1].h], axis=-1)

    return output_feature

  def encode(self, input_data):
    """
    Args:
      input_data: dict of tensor
    Returns:
      encode_output: [batch_size, out_dim]
    """
    #encode_history
    with tf.variable_scope('history') as scope:
      history_feature = self.history_encode(input_data, self.params['history_encoder_params'])    
    #[batch_size, max_q_len, word_dim]
    question = input_data['q']
    #[batch_size, max_qa_len, max_q_len, word_dim]
    questions = input_data['h_q']
    batch_size = tf.shape(questions)[0]
    max_q_len = tf.shape(question)[1]
    max_qa_len = tf.shape(questions)[1]
    new_max_q_len = tf.shape(questions)[2]
    if self.params['video']:
      self.params['video_encoder_params']['is_train'] = self.params['is_train']
      with tf.variable_scope('video') as scope:
        #[batch_size, video_feature_dim]
        video_feature = self.video_encode(input_data, self.params['video_encoder_params'])
      question_video_feature = tf.tile(tf.expand_dims(video_feature, axis=1), multiples=[1, max_q_len, 1])
      questions_video_feature = tf.tile(video_feature[:, None, None, :], multiples=[1, max_qa_len, new_max_q_len, 1])
      question = tf.concat([question, question_video_feature], axis=2)
      questions = tf.concat([questions, questions_video_feature], axis=3)
    with tf.variable_scope('questions_and_question') as scope:
      question_len = input_data['q_len']
      questions_len = input_data['h_q_len']
      questions_feature, question_feature = self.qs_q_encode(questions, questions_len, question, question_len, self.params['question_encoder_params'])
    #construction attention mechanism
    attention_class = eval(self.params['attention_params']['class'])
    query_dim = int(questions_feature.get_shape()[-1])
    self.atm = attention_class(query_dim, history_feature,
                               memory_sequence_length=input_data['h_len'],
                               probability_fn=tf.identity)
    atm_initial = self.atm.initial_alignments(batch_size, dtype=tf.float32)
    #
    merge_dim = int(history_feature.get_shape()[-1]) + \
            int(questions_feature.get_shape()[-1])

    #history: [batch_size, max_qa_len, history_dim]
    #questions: [batch_size, max_qa_len, question_dim]
    #question: [batch_size, question_dim]
    #h_len: [batch_size]
    memory_features = {'history': history_feature,
                       'questions': questions_feature,
                       'question': question_feature,
                       'h_len': input_data['h_len'],
                       'batch_size': batch_size,
                       'max_qa_len': max_qa_len,
                       'atm': self.atm,
                       'merge_dim': merge_dim,
                       'atm_initial': atm_initial}

    #batch_size loop
    init_batch_id = 0
    #[batch_size, max_qa_len, merge_dim]
    init_batch_merged_features = tf.TensorArray(dtype=tf.float32, size=batch_size, clear_after_read = False)

    def is_batch_end(batch_id, batch_merged_features):
      return batch_id < memory_features['batch_size']
    def write_cur_batch_id_info(batch_id, batch_merged_features):
      init_time_step = 0
      init_merged_features = tf.TensorArray(dtype=tf.float32, size=max_qa_len, clear_after_read = False)
      def is_seq_end(time_step, merged_features):
        return time_step < max_qa_len
      def write_cur_step_info(time_step, merged_features):
        def get_cur_attention(query):
          """
          query: [question_dim]
          """
          raw_query = query
          atm = memory_features['atm']
          atm_initial = memory_features['atm_initial']
          query = tf.tile(query[None, :], multiples=[memory_features['batch_size'], 1])
          score = atm(query, atm_initial)[batch_id, :]
          mask = tf.sequence_mask([time_step], memory_features['max_qa_len'])[0, :]
          score_mask_values = float('-inf') * tf.ones_like(score)
          new_score = tf.where(mask, score, score_mask_values)
          #normalize rate [max_qa_len]
          align = tf.nn.softmax(new_score)
          atten_history = tf.reduce_sum(memory_features['history'][batch_id, :, :] * align[:, None], axis=0)
          return tf.concat([atten_history, raw_query], 0)
        def f1():
          """pre-history"""
          result = tf.cond(time_step > 0, f5, f6)
          return result
        def f3():
          """final question"""
          result = tf.cond(time_step > 0, f7, f6)
          return result
        def f4():
          """zero feature"""
          return tf.zeros([memory_features['merge_dim']])
        def f2():
          """after pre-history"""
          result = tf.cond(time_step > memory_features['h_len'][batch_id], f4, f3)
          return result
        def f5():
          query = memory_features['questions'][batch_id, time_step, :]
          return get_cur_attention(query)
        def f6():
          """time == 0"""
          query = memory_features['questions'][batch_id, time_step, :]
          atten_history = tf.zeros([int(memory_features['history'].get_shape()[-1])])
          return tf.concat([atten_history, query], 0)
        def f7():
          """final not 0 question"""
          query = memory_features['question'][batch_id, :]
          return get_cur_attention(query)

        cur_feature = tf.cond(time_step < memory_features['h_len'][batch_id], f1, f2)
        merged_features = merged_features.write(time_step, cur_feature)
        return time_step + 1, merged_features
      _, merged_features = tf.while_loop(is_seq_end, write_cur_step_info, [init_time_step, init_merged_features])
      #[max_qa_len, merge_dim]
      merged_features = merged_features.stack()
      batch_merged_features = batch_merged_features.write(batch_id, merged_features)
      return batch_id + 1, batch_merged_features

    _, batch_merged_features = tf.while_loop(is_batch_end, write_cur_batch_id_info, [init_batch_id, init_batch_merged_features])
    #[batch_size, max_qa_len, merge_dim]
    batch_merged_features = batch_merged_features.stack()

    batch_merged_features_len = input_data['h_len'] + 1

    #encode_dialog
    with tf.variable_scope('dialog') as scope:
      dialog_feature = self.dialog_encode(batch_merged_features, batch_merged_features_len, 
                                           self.params['dialog_encoder_params'])

    encode_output = dialog_feature
    return encode_output
