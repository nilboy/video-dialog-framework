from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from mvd import GraphModule
from mvd.encoder.video_encoder import *

from pydoc import locate
__all__ = [
  'VideoEncoder',
  'construct_video_encoder',
]
@six.add_metaclass(abc.ABCMeta)
class VideoEncoder(GraphModule):
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
  def encode(self, video_input_data):
    """
    Args:
      video_input_data: video_feature tensor [batch_size, frame_size, feature_size]
    Returns:
      video_encode_output: tensor [batch_size, feature_num]
    """
    raise NotImplementedError

def construct_video_encoder(params, name):
  encoder_class = locate(params['class'])
  params['params']['is_train'] = params['is_train']
  return encoder_class(params['params'], name)
