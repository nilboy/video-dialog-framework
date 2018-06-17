from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mvd.net import Net
from mvd.ops import *
from mvd.encoder import *
from mvd.decoder import *

class MNet(Net):

  def __init__(self, net_params):
    self.phase = net_params['phase']
    self.style = net_params['style']
    self.vgg_feature_num = net_params['vgg_feature_num']
    self.c3d_feature_num = net_params['c3d_feature_num']
    self.candidate_num = net_params['candidate_num']
    self.embedding_file = net_params['embedding_file']
    self.vocab_file = net_params['vocab_file']
    self.encoder_params = net_params['encoder_params']
    self.decoder_params = net_params['decoder_params']
    #add phase to decoder
    self.decoder_params['params']['phase'] = self.phase
    
  @property
  def input_data(self):
    return self._input_data

  @property
  def output_data(self):
    return self._output_data

  @property
  def loss(self):
    """scalar float32"""
    return self._loss

  def construct_graph(self, is_train=True):
    embedding_matrix = get_embedding_matrix(self.embedding_file, name='embedding')
    processed_input_data = input_process(self.input_data, self.vocab_file, embedding_matrix)
    self.encoder_params['params']['is_train'] = is_train
    self.decoder_params['params']['is_train'] = is_train
    encoder = construct_encoder(self.encoder_params, 'encoder')
    decoder = construct_decoder(self.decoder_params, 'decoder')
    encode_output = encoder(processed_input_data)
    decode_output = decoder(processed_input_data, encode_output)
    return decode_output

  def construct_mc_loss(self, logits):
    batch_size = tf.shape(logits)[0]
    labels = tf.zeros([batch_size], dtype=tf.int32)  
    self._loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  def make_placeholder(self):
    #construct input_data placeholders
    self._input_data = {}
    self._input_data['vgg'] = tf.placeholder(tf.float32, [None, None, self.vgg_feature_num])
    self._input_data['c3d'] = tf.placeholder(tf.float32, [None, None, self.c3d_feature_num])
    self._input_data['h_q'] = tf.placeholder(tf.string, [None, None])
    self._input_data['h_a'] = tf.placeholder(tf.string, [None, None])
    self._input_data['h_len'] = tf.placeholder(tf.int32, [None])
    self._input_data['q'] = tf.placeholder(tf.string, [None])
    self._input_data['a'] = tf.placeholder(tf.string, [None])
    self._input_data['candidate_a'] = tf.placeholder(tf.string, [None, self.candidate_num])    

  def build_mc_model(self):
    self.make_placeholder()
    #construct_model
    initializer = tf.random_uniform_initializer(-1e-2, 1e-2)
    if self.phase == 'train':
      with tf.name_scope('train'):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          decode_output = self.construct_graph(True)
          self.construct_mc_loss(decode_output)
      with tf.name_scope('evaluate'):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          evaluate_logits = self.construct_graph(False)
          self._output_data = tf.nn.softmax(evaluate_logits)
      tf.summary.scalar('loss', self.loss)
    else:
      with tf.name_scope('evaluate'):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          evaluate_logits = self.construct_graph(False)
          self._output_data = tf.nn.softmax(evaluate_logits)

  def construct_oe_loss(self, decode_output):
    """
      Args:
        decode_output: dict{'logits', 'a_len', 'label_a'}
    """
    logits = decode_output['logits']
    max_time = tf.reduce_max(decode_output['a_len'])
    # [batch_size, max_time]
    target_output = decode_output['label_a'][:, 0:max_time]
    batch_size = tf.shape(target_output)[0]
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        decode_output['a_len'], max_time, dtype=logits.dtype)
    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(batch_size)
    self.nilboy = tf.shape(target_weights)
    return loss


  def make_oe_output(self, decode_output):
    """
    Return:
      dict {'answer'}
    """
    output_data = {}
    #
    reverse_out_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(vocabulary_file=self.vocab_file,
                                           default_value='<UNK>',
                                           name='reverse_out_vocab_table'
                                          )
    output_words = reverse_out_vocab_table.lookup(tf.to_int64(decode_output['sample_id']))
    output_string = tf.reduce_join(output_words, axis=1, separator=' ')
    output_data['answer'] = output_string
    return output_data


  def build_oe_model(self):
    self.make_placeholder()
    #construct_model
    initializer = tf.random_uniform_initializer(-1e-2, 1e-2)
    if self.phase == 'train':
      with tf.name_scope('train'):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          decode_output = self.construct_graph(True)
          self._loss = self.construct_oe_loss(decode_output)
      with tf.name_scope('evaluate'):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          evaluate_decode_output = self.construct_graph(False)
          self._output_data = {}
          self._output_data['loss'] = self.construct_oe_loss(evaluate_decode_output)
      tf.summary.scalar('loss', self.loss)
    else:
      with tf.name_scope('evaluate'):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          evaluate_decode_output = self.construct_graph(False)
          self._output_data = self.make_oe_output(evaluate_decode_output) 

  def build_model(self):
    if self.style == 'MC':
      self.build_mc_model()
    else:
      self.build_oe_model()