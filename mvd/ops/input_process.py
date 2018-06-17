from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import cPickle as pkl
import sys
is_py2 = sys.version[0] == '2'

def length(input_tensor):
  """
  Args:
    input_tensor: [..., max_len]
  Returns:
    input_length: [...] tf.int32
  """
  mask = input_tensor > 0
  mask = tf.cast(mask, tf.int32)
  return tf.reduce_sum(mask, axis=-1)

def str2id(input_str, vocab_table, embedding_matrix):
  """
  Args:
    input_str: [...]  tf.string
    vocab_table: HashTable
    embedding_matrix
  Returns:
    output_embed: [..., max_len, word_dim] tf.int
    output_len:[...] tf.int
  """
  input_shape = tf.shape(input_str)
  input_str = tf.reshape(input_str, [-1])
  #add <s>, </s>
  input_str = '<s> ' + input_str + ' </s>'
  #split
  split_input_str = tf.string_split(input_str)
  #sparse tensor
  input_id = vocab_table.lookup(split_input_str)
  #convert2dense
  output_id = tf.sparse_tensor_to_dense(input_id)

  output_id = tf.reshape(output_id, tf.concat([input_shape, tf.shape(output_id)[-1:]], 0))
  output_len = length(output_id)
  output_embed = tf.nn.embedding_lookup(embedding_matrix, output_id)
  return output_embed, output_id, output_len
def get_embedding_matrix(embedding_file, name=None):
  """
  Args:
    embedding_file: path of embedding_file
  Returns:
    embedding_matrix: [vocab_size + 4, word_dim]
  """
  with tf.variable_scope(name or tf.get_variable_scope()) as scope:
    if is_py2:
      embed1 = pkl.load(open(embedding_file, 'rb'))
    else:
      embed1 = pkl.load(open(embedding_file, 'rb'), encoding='bytes')
    _, word_dim = embed1.shape
    embed2 = tf.get_variable('part_embedding', shape=[4, word_dim], dtype=tf.float32)
    embed1 = tf.constant(embed1, dtype=tf.float32)
    embedding_matrix = tf.concat([embed2, embed1], axis=0)
    return embedding_matrix


def input_process(input_data, vocab_file, embedding_matrix):
  """
    Args:
      intput_data:
      {
        'vgg':  [batch_size, vgg_frames, vgg_feature_num] tf.float32,
        'c3d':  [batch_size, c3d_frames, c3d_feature_num] tf.float32,
        'h_q':  [batch_size, max_qa_len] tf.string,
        'h_a':  [batch_size, max_qa_len] tf.string.
        'h_len': [batch_size] tf.int32
        'q':     [batch_size] tf.string
        'a'    : [batch_size] tf.string
        'candidate_a': [batch_size, candidate_num] tf.string 
      } 
      vocab_file: path of file
      embedding_matrix: tf.tensor [vacab_size * embedding_size]
    Returns:
      {
        'vgg': [batch_size, vgg_frames, vgg_feature_num] tf.float32,
        'c3d': [batch_size, c3d_frames, c3d_feature_num] tf.float32
        'h_qa': [batch_size, max_qa_len, max_len, word_dim] tf.int
        'h_qa_len': [batch_size, max_qa_len] tf.int
        'h_all': [batch_size, max_len, word_dim]
        'h_all_len': [batch_size]
        'h_q': [batch_size, max_qa_len, max_q_len, word_dim] tf.int,
        'h_q_len': [batch_size, max_qa_len] tf.int
        ***'h_a': [batch_size, max_qa_len, max_a_len, word_dim] tf.int,
        ***'h_a_len': [batch_size, max_qa_len]
        'h_len': [batch_size] tf.int32
        'q': [batch_size, max_q_len, word_dim]   tf.int
        'q_len': [batch_size] tf.int
        'candidate_a': [batch_size, candidate_num, max_a_len, word_dim] tf.int
        'candidate_a_len': [batch_size, candidate_num] tf.int
        'embedding_matrix': tf.tensor [vocab_size, embedding_size]
      }
  """
  new_input_data = {}
  vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocab_file,
                                           default_value=1,
                                           name='vocab_table'
                                          )
  #vgg, c3d, h_len
  new_input_data['vgg'] = input_data['vgg']
  new_input_data['c3d'] = input_data['c3d']
  new_input_data['h_len'] = input_data['h_len']
  #
  
  h_q, _, h_q_len = str2id(input_data['h_q'], vocab_table, embedding_matrix)
  new_input_data['h_q'] = h_q
  new_input_data['h_q_len'] = h_q_len
  """
  #
  h_a, h_a_len = str2id(input_data['h_a'], vocab_table, embedding_matrix)
  new_input_data['h_a'] = h_a
  new_input_data['h_a_len'] = h_a_len
  """
  h_qa, _, h_qa_len = str2id(input_data['h_q'] + ' ' + input_data['h_a'], vocab_table, embedding_matrix)

  new_input_data['h_qa'] = h_qa
  new_input_data['h_qa_len'] = h_qa_len
  #h_all
  h_all = tf.reduce_join(input_data['h_q'] + ' ' + input_data['h_a'], 1, separator=' ')
  h_all, _, h_all_len = str2id(h_all, vocab_table, embedding_matrix)
  new_input_data['h_all'] = h_all
  new_input_data['h_all_len'] = h_all_len
  #
  q, _, q_len = str2id(input_data['q'], vocab_table, embedding_matrix)
  new_input_data['q'] = q
  new_input_data['q_len'] = q_len
  #
  candidate_a, candidate_a_id, candidate_a_len = str2id(input_data['candidate_a'], vocab_table, embedding_matrix)
  new_input_data['candidate_a'] = candidate_a
  new_input_data['candidate_a_len'] = candidate_a_len
  new_input_data['a_id'] = candidate_a_id[:, 0, 1:]
  new_input_data['embedding_matrix'] = embedding_matrix

  return new_input_data
