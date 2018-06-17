from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from mvd import GraphModule
from mvd.decoder import *
import tensorflow as tf
from pydoc import locate

__all__ = [
  'Decoder',
  'Basicdecoder',
  'construct_decoder',
]

@six.add_metaclass(abc.ABCMeta)
class Decoder(GraphModule):
  """Decoder abstract interface object"""

  def __init__(self, params, name):
    GraphModule.__init__(self, name)
    self._params = params

  @property
  def params(self):
    return self._params

  def _build(self, inputs, *args, **kwargs):
    return self.decode(inputs, *args, **kwargs)

  @abc.abstractmethod
  def decode(self, input_data, encode_output):
    """
    Args:
      input_data: dict of tensor
      encode_output: tensor [batch_size, feature_num]
    Returns:
      decode_output: tensor [batch_size, candidate_num] tf.float32
    """
    raise NotImplementedError


class Basicdecoder(Decoder):

  def decode(self, input_data, encode_output):
    input_data = input_data['candidate_a'][:, :, 1, :]
    encode_output = encode_output[:, None, :]
    return tf.reduce_sum(input_data * encode_output, axis=2)

def construct_decoder(params, name):
  decoder_class = locate(params['class'])
  return decoder_class(params['params'], name)
