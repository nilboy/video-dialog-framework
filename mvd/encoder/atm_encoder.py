from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder import *
from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'ATMEncoder',
]

class ATMEncoder(Encoder):
  """Attention-Memory Encoder
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


  def bi_attention(self, source1, source2, source1_len=None, source2_len=None, scope=None, params=None):
    """
    Args:
      source1: [batch_size, max_len1, source1_dim] tf.float32,
      source2: [batch_size, max_len2, source2_dim] tf.float32,
      source1_len: [batch_size] tf.int32,
      source2_len: [batch_size] tf.int32
    """
    source1_dim = int(source1.get_shape()[-1])
    source2_dim = int(source2.get_shape()[-1])
    with tf.variable_scope(scope or tf.get_variable_scope()) as scope:
      #source1 Attention to source2
      with tf.variable_scope('source1') as scope:
        cells = self.rnn_cells(params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(params['attention_params']['class'])

          atm = attention_class(source1_dim, source2, memory_sequence_length=source2_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if params['cell_params']['stack_type'] == 'stack':
          cell = tf.contrib.rnn.MultiRNNCell(cells)
          """
          outputs, _ = tf.nn.dynamic_rnn(cell, source1,
                                              sequence_length=source1_len, dtype=tf.float32)
          """
          outputs, _ = tf.nn.dynamic_rnn(cell, source1, dtype=tf.float32)
        else:
          """
          outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source1, 
                                          sequence_length=source1_len, dtype=tf.float32)
          """
          outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source1, 
                                          dtype=tf.float32)
          outputs = tf.concat(outputs, -1)

        #[batch_size, max_len, source1_1_dim]
        source1_1 = outputs

      #source2 Attention to source1
      with tf.variable_scope('source2') as scope:
        cells = self.rnn_cells(params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(params['attention_params']['class'])
          atm = attention_class(source2_dim, source1, memory_sequence_length=source1_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if params['cell_params']['stack_type'] == 'stack':
          cell = tf.contrib.rnn.MultiRNNCell(cells)
          #questions
          """
          outputs, _ = tf.nn.dynamic_rnn(cell, source2,
                                              sequence_length=source2_len, dtype=tf.float32)
          """
          outputs, _ = tf.nn.dynamic_rnn(cell, source2,
                                              dtype=tf.float32)

        else:
          outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source2, 
                                          dtype=tf.float32)
          outputs = tf.concat(outputs, -1)
        #[batch_size, max_len, source1_1_dim]
        source2_1 = outputs
      #
      source1_1_dim = int(source1_1.get_shape()[-1])
      source2_1_dim = int(source2_1.get_shape()[-1])
      #source1_1 Attention to source2_1
      with tf.variable_scope('source1_1') as scope:
        cells = self.rnn_cells(params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(params['attention_params']['class'])
          atm = attention_class(source1_1_dim, source2_1, memory_sequence_length=source2_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if params['cell_params']['stack_type'] == 'stack':
          cell = tf.contrib.rnn.MultiRNNCell(cells)
          #questions
          """
          outputs, state = tf.nn.dynamic_rnn(cell, source1_1,
                                              sequence_length=source1_len, dtype=tf.float32)
          """
          outputs, state = tf.nn.dynamic_rnn(cell, source1_1,
                                              dtype=tf.float32)
        else:
          """
          outputs, state = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source1_1, 
                                          sequence_length=source1_len, dtype=tf.float32)
          """
          outputs, state = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source1_1, 
                                                    dtype=tf.float32)
        output_states = []
        for state_item in state:
          output_states.append(state_item.cell_state.h)
        output_feature = tf.concat(output_states, axis=-1)
        #[batch_size, max_len, source1_1_dim]
        source1_2 = output_feature

      #source2_1 Attention to source1_1
      with tf.variable_scope('source2_1') as scope:
        cells = self.rnn_cells(params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(params['attention_params']['class'])
          atm = attention_class(source2_1_dim, source1_1, memory_sequence_length=source1_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if params['cell_params']['stack_type'] == 'stack':
          cell = tf.contrib.rnn.MultiRNNCell(cells)
          #questions
          """
          outputs, _ = tf.nn.dynamic_rnn(cell, source2_1,
                                              sequence_length=source2_len, dtype=tf.float32)
          """
          outputs, state = tf.nn.dynamic_rnn(cell, source2_1,
                                              dtype=tf.float32)
        else:
          """
          outputs, _ = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source2_1, 
                                          sequence_length=source2_len, dtype=tf.float32)
          """
          outputs, state = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], source2_1, 
                                          dtype=tf.float32)

        output_states = []
        for state_item in state:
          output_states.append(state_item.cell_state.h)
        output_feature = tf.concat(output_states, axis=-1)
        #[batch_size, max_len, source1_1_dim]
        source2_2 = output_feature

      return source1_2, source2_2

  def fusion(self, source1, source2, question1, question2, scope=None):
    """
    Args:
      source1: [batch_size, source1_dim]
      source2: [batch_size, source2_dim]
      question1: [batch_size, query_dim1]
      question2: [batch_size, query_dim2]
    Returns:
      result: [batch_size, output_dim]
    """
    
    with tf.variable_scope(scope or tf.get_variable_scope()) as scope:
      query_dim_1 = int(question1.get_shape()[-1])
      score1 = tf.layers.dense(tf.concat([source1, question1], -1),
                      query_dim_1,
                      activation=tf.nn.tanh,
                      name='source1_fusion_1')
      score1 = tf.layers.dense(score1, 1, use_bias=False, name="source1_fusion_2")
      query_dim_2 = int(question2.get_shape()[-1])
      score2 = tf.layers.dense(tf.concat([source2, question2], -1),
                      query_dim_2,
                      activation=tf.nn.tanh,
                      name='source1_fusion_2')
      score2 = tf.layers.dense(score2, 1, use_bias=False, name="source2_fusion_2")
      align = tf.nn.softmax(tf.concat([score1, score2], -1))
      result = tf.reduce_sum(tf.concat([source1[:, :, None], source2[:, :, None]], -1) * align[:, None, :], -1)
      return result


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

  def memory_attention(self, query, memory, memory_length, reuse=None):
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

    reshape_atten_memory = tf.layers.dense(atten_memory, out_dim, name='atten_memory_reshape', reuse=reuse)

    return reshape_atten_memory + query

  def encode(self, input_data):
    """
    Args:
      input_data: dict of tensor
    Returns:
      encode_output: [batch_size, out_dim]
    """
    qv_resized_dim = self.params['encode_out_dim']
    if self.params['history']:
      with tf.variable_scope('history_encode') as scope:
        #[batch_size, max_qa_len, history_dim]
        history_feature = self.history_encode(input_data, self.params['history_encoder_params'])
        history_len = input_data['h_len']
        qv_resized_dim = int(history_feature.get_shape()[-1])

    question_feature = input_data['q']
    question_len = input_data['q_len']
    sources = []
    if self.params['video']:
      if self.params['vgg']:
        #vgg
        batch_size = tf.shape(input_data['vgg'])[0]
        vgg_frames = tf.shape(input_data['vgg'])[1]
        vgg_len = tf.ones([batch_size], dtype=tf.int32) * vgg_frames
        sources.append(['vgg', input_data['vgg'], vgg_len])
      if self.params['c3d']:
        #c3d
        batch_size = tf.shape(input_data['c3d'])[0]
        c3d_frames = tf.shape(input_data['c3d'])[1]
        c3d_len = tf.ones([batch_size], dtype=tf.int32) * c3d_frames
        sources.append(['c3d', input_data['c3d'], c3d_len])
    merge_features = []
    for source_item in sources:
      a, b = self.bi_attention(source_item[1], question_feature, 
                        source1_len=source_item[2], source2_len=question_len, scope=source_item[0],
                                     params=self.params['qvideo_encoder_params'])
      merge_features.append(a)
      merge_features.append(b)
    fusion_feature = tf.concat(merge_features, -1)
    #[batch_size, encode_out_dim]
    qv_feature = tf.layers.dense(fusion_feature, qv_resized_dim, name='encode_reshape')
    #
    query = qv_feature
    #
    if self.params['history']:
      for i in range(self.params['memory_hop_num']):
        if i == 0:
          query = self.memory_attention(query, history_feature, history_len, reuse=False)
        else:
          query = self.memory_attention(query, history_feature, history_len, reuse=True)
      query = tf.layers.dense(query, self.params['encode_out_dim'], name='query_reshape')
    return query