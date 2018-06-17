from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder import *
from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'MATEncoder',
]

class MATEncoder(Encoder):
  """Multi-Attention Encoder
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


  def bi_attention(self, source1, source2, source1_len=None, source2_len=None, scope=None):
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
        cells = self.rnn_cells(self.params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(self.params['attention_params']['class'])

          atm = attention_class(source1_dim, source2, memory_sequence_length=source2_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=self.params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if self.params['cell_params']['stack_type'] == 'stack':
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
        cells = self.rnn_cells(self.params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(self.params['attention_params']['class'])
          atm = attention_class(source2_dim, source1, memory_sequence_length=source1_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=self.params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if self.params['cell_params']['stack_type'] == 'stack':
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
        cells = self.rnn_cells(self.params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(self.params['attention_params']['class'])
          atm = attention_class(source1_1_dim, source2_1, memory_sequence_length=source2_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=self.params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if self.params['cell_params']['stack_type'] == 'stack':
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
        cells = self.rnn_cells(self.params['cell_params'])
        new_cells = []
        for index, cell in enumerate(cells):
          attention_class = eval(self.params['attention_params']['class'])
          atm = attention_class(source2_1_dim, source1_1, memory_sequence_length=source1_len)
          at_cell = tf.contrib.seq2seq.AttentionWrapper(cell, atm,
                                attention_layer_size=self.params['attention_params']['num_units'])
          new_cells.append(at_cell)
        cells = new_cells

        if self.params['cell_params']['stack_type'] == 'stack':
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

  def history_encode(self, input_data):
    #encode_history
    #[batch_size, max_qa_len, max_q_len, word_dim]
    params = self.params['cell_params']
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

  def encode(self, input_data):
    """
    Args:
      input_data: dict of tensor
    Returns:
      encode_output: [batch_size, out_dim]
    """
    with tf.variable_scope('history_encode') as scope:
      #[batch_size, max_qa_len, history_dim]
      history_feature = self.history_encode(input_data)
      history_len = input_data['h_len']
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
    if self.params['history']:
      #history
      sources.append(['history', history_feature, history_len])
    merge_features = []
    for source_item in sources:
      a, b = self.bi_attention(source_item[1], question_feature, 
                        source1_len=source_item[2], source2_len=question_len, scope=source_item[0])
      merge_features.append(a)
      merge_features.append(b)
    fusion_feature = tf.concat(merge_features, -1)
    #reshape dims
    encode_out_dim = self.params['encode_out_dim']
    encode_output = tf.layers.dense(fusion_feature, encode_out_dim, name='encode_reshape')
    return encode_output