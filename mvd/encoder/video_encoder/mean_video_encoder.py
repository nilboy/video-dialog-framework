from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mvd.encoder.video_encoder import *

import tensorflow as tf

__all__ = [
  'MeanVideoEncoder',
]

class MeanVideoEncoder(VideoEncoder):
  """MeanVideo Encoder
  """

  def encode(self, video_input_data):
    """
    Args:
      video_input_data: [batch_size, frame_size, frame_feature_num]
    Returns:
      video_encode_out: [batch_size, video_output_dim]
    """
    video_feature = tf.reduce_mean(video_input_data, axis=1)

    if self.params['is_train']:
      video_feature = tf.layers.dropout(video_feature, rate=1.0 - self.params['dropout_keep_rate'])

    return video_feature

