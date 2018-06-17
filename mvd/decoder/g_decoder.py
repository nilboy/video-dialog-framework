from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.decoder import *

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

__all__ = [
  'GDecoder',
]

class GDecoder(Decoder):
  """generative Decoder
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

  def _build_deocder_init_state(self, encode_output):
    #
    encode_init_state = []
    for i in range(self.params['layer_num']):
      split_encode_output = encode_output[:, i * self.params['num_units']:(i + 1) * self.params['num_units']]
      encode_init_state.append(tf.contrib.rnn.LSTMStateTuple(split_encode_output, split_encode_output))
    encode_output = tuple(encode_init_state)
    #
    if self.params['phase'] == 'infer' and self.params['beam_width'] > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encode_output, multiplier=self.params['beam_width'])
    else:
      decoder_initial_state = encode_output

    return decoder_initial_state

  def _build_decoder_cell(self, encode_output):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    cells = self.rnn_cells()
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    decoder_initial_state = self._build_deocder_init_state(encode_output)

    return cell, decoder_initial_state

  def decode_train(self, input_data, encode_output):
    tgt_sos_id = tf.cast(2, tf.int32)
    tgt_eos_id = tf.cast(3, tf.int32)
    cell, decoder_initial_state = self._build_decoder_cell(encode_output)
    #[batch_size, max_a_len, word_dim]
    answer = input_data['candidate_a'][:, 0, :, :]
    answer_len = input_data['candidate_a_len'][:, 0]
    decoder_emb_inp = answer
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, answer_len - 1)
    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
    # Dynamic decoding
    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                          my_decoder,
                                          swap_memory=True)
    sample_id = outputs.sample_id
    with tf.variable_scope("decoder") as scope:
      logits = self.output_layer(outputs.rnn_output)
    decode_output = {}
    decode_output['logits'] = logits
    decode_output['a_len'] = answer_len - 1
    decode_output['label_a'] = input_data['a_id']
    return decode_output

  def decode_infer(self, input_data, encode_output):
    batch_size = tf.shape(input_data['candidate_a'])[0]
    tgt_sos_id = tf.cast(2, tf.int32)
    tgt_eos_id = tf.cast(3, tf.int32)
    maximum_iterations = 30
    cell, decoder_initial_state = self._build_decoder_cell(encode_output)
    beam_width = self.params['beam_width']
    start_tokens = tf.fill([batch_size], tgt_sos_id)
    end_token = tgt_eos_id
    if beam_width > 0:
      my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          cell=cell,
          embedding=input_data['embedding_matrix'],
          start_tokens=start_tokens,
          end_token=end_token,
          initial_state=decoder_initial_state,
          beam_width=beam_width,
          output_layer=self.output_layer)
    else:
      # Helper
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(input_data['embedding_matrix'], start_tokens, end_token)

      # Decoder
      my_decoder = tf.contrib.seq2seq.BasicDecoder(
          cell,
          helper,
          decoder_initial_state,
          output_layer=self.output_layer  # applied per timestep
      )
    # Dynamic decoding
    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        my_decoder,
        maximum_iterations=maximum_iterations,
        swap_memory=True)
    if beam_width > 0:
      logits = tf.no_op()
      sample_id = outputs.predicted_ids[:, :, 0]
    else:
      logits = outputs.rnn_output
      sample_id = outputs.sample_id
    decode_output = {}
    decode_output['sample_id'] = sample_id
    return decode_output


  def decode(self, input_data, encode_output):
    """
    Args:
      input_data: dict of tensor
      encode_output: [batch_size, encode_out_dims]
    Returns:
      logits: [batch_size, candidate_num]
    """
    tgt_vocab_size = int(input_data['embedding_matrix'].shape[0])
    self.output_layer = layers_core.Dense(tgt_vocab_size, use_bias=False, name="output_vocab_projection")
    if self.params['phase'] == 'train':
      return self.decode_train(input_data, encode_output)
    else:
      return self.decode_infer(input_data, encode_output)
