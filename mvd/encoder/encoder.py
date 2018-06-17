from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from mvd import GraphModule
from mvd.encoder import *

from pydoc import locate

__all__ = [
  'Encoder',
  'BasicEncoder',
  'construct_encoder',
]

@six.add_metaclass(abc.ABCMeta)
class Encoder(GraphModule):
  """Encoder abstract interface object"""

  def __init__(self, params, name):
    GraphModule.__init__(self, name)
    self._params = params

  @property
  def params(self):
    return self._params

  def _build(self, inputs, *args, **kwargs):
    return self.encode(inputs, *args, **kwargs)

  @abc.abstractmethod
  def encode(self, input_data):
    """
    Args:
      input_data: dict of tensor
    Returns:
      encode_output: tensor [batch_size, feature_num]
    """
    raise NotImplementedError


class BasicEncoder(Encoder):

  def encode(self, input_data):
    return input_data['vgg'][:, 0, 0:100]

def construct_encoder(params, name):
  encoder_class = locate(params['class'])
  return encoder_class(params['params'], name)
