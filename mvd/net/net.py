from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

@six.add_metaclass(abc.ABCMeta)
class Net(object):
  """An net abstract interface object."""

  @abc.abstractproperty
  def input_data(self):
    raise NotImplementedError

  @abc.abstractproperty
  def output_data(self):
    raise NotImplementedError

  @abc.abstractproperty
  def loss(self):
    """scalar float32"""
    raise NotImplementedError

  @abc.abstractmethod
  def build_model(self):
    """build model graph"""
    raise NotImplementedError