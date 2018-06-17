"""A class of DataLoader """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import sys
from six.moves import cPickle as pkl
from six import next
#import cPickle as pkl
import redis
import numpy as np
import math
import time
is_py2 = sys.version[0] == '2'
if is_py2:
  from Queue import Queue
else:
  from queue import Queue
#from multiprocessing import Queue
#from multiprocessing import Process as Thread
from threading import Thread

__all__ = [
  'BasicDataLoader',
]

class BasicDataLoader(object):

  def __init__(self, data_loader_params):
    self.BATCH_QUEUE_MAX = 100
    self.batch_size = data_loader_params['batch_size']
    self.video_frames = data_loader_params['video_frames']
    self.max_qa_len = data_loader_params['max_qa_len']
    self.vgg_frames = self.video_frames
    self.c3d_frames = self.video_frames - 15
    self.num_records = data_loader_params['num_records']
    self.candidate_num = data_loader_params['candidate_num']
    self.is_shuffle = data_loader_params['is_shuffle']
    self.record_point = 0
    self.port = data_loader_params['port']
    self.mode = data_loader_params['mode']
    self.records_ids = range(0, self.num_records)
    self.rdb = redis.StrictRedis(host='localhost', port=self.port, db=0)
    #
    self.num_record_queues = 1
    self._num_example_q_threads = self.num_record_queues * 8 # num threads to fill record queue
    self._num_batch_q_threads = self.num_record_queues * 4 # num threads to fill batch queue

    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
    self._record_queues = [Queue(int(self.BATCH_QUEUE_MAX * self.batch_size / self.num_record_queues)) for _ in range(self.num_record_queues)]

    self._example_q_threads = []
    for index in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_record_queue, args=(index,)))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for index in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue, args=(index % self.num_record_queues,)))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()


  @property
  def batch_num_per_epoch(self):
    return int(math.floor(self.num_records / self.batch_size))

  def reset(self):
    if self.is_shuffle:
      random.shuffle(self.records_ids)
    self.record_point = 0

  def record_id_generator(self, thread_id):
    while True:
      per_thread_nums = int(self.num_records / self._num_example_q_threads)
      if thread_id == self._num_example_q_threads - 1:
        records_index = range(thread_id * per_thread_nums, self.num_records)
      else:
        records_index = range(thread_id * per_thread_nums, (thread_id + 1) * per_thread_nums)
      if self.is_shuffle:
        random.shuffle(records_index)
      for index in records_index:
        yield index

  def fill_record_queue(self, thread_id):
    record_queue_id = int(thread_id % self.num_record_queues)
    record_index_gen = self.record_id_generator(thread_id)
    while True:
      record_id = next(record_index_gen)
      record = self.rdb.get(self.mode + str(record_id))
      record = pkl.loads(record)
      #
      input_data = {}
      video_feature = pkl.loads(self.rdb.get(record['video_feature']))
      input_data['vgg'] = video_feature['vgg'].astype(np.float32)
      input_data['c3d'] = video_feature['c3d'].astype(np.float32)

      temp_h_q = [''] * self.max_qa_len
      temp_h_a = [''] * self.max_qa_len
      dialog = record['dialog']
      dialog_len = len(dialog)
      input_data['qa_len'] = dialog_len + 1
      for i in range(0, dialog_len - 1):
        q = dialog[i][0]
        a =  pkl.loads(self.rdb.get('a' + str(dialog[i][1])))[0]
        temp_h_a[i + 1] = a
        temp_h_q[i + 1] = q
      q = dialog[dialog_len - 1][0]
      a =  pkl.loads(self.rdb.get('a' + str(dialog[dialog_len - 1][1])))[0]
      input_data['h_q'] = temp_h_q
      input_data['h_a'] = temp_h_a
      input_data['h_len'] = dialog_len
      input_data['q'] = q
      input_data['a'] = a
      #
      candidate_id = pkl.loads(self.rdb.get('a' + str(dialog[dialog_len - 1][1])))[1]
      temp_candidate_answer = []
      for index in candidate_id:
        temp_candidate_answer.append(pkl.loads(self.rdb.get('a' + str(index)))[0])
      input_data['candidate_a'] = temp_candidate_answer
      #random record_queue_id
      #record_queue_id = random.randint(0, self.num_record_queues - 1)
      self._record_queues[record_queue_id].put(input_data) 

  def fill_batch_queue(self, record_queue_id):
    while True:
      record_list = []
      for i in range(self.batch_size):
        record_list.append(self._record_queues[record_queue_id].get())
      input_data = {'vgg': [], 'c3d': [], 'h_q': [], 'h_a': [], 'h_len': [], 'q': [], 'a': [], 'candidate_a': []}
      new_max_qa_len = 0

      for record in record_list:      
        input_data['vgg'].append(record['vgg'])
        input_data['c3d'].append(record['c3d'])
        input_data['h_q'].append(record['h_q'])
        input_data['h_a'].append(record['h_a'])
        input_data['h_len'].append(record['h_len'])
        input_data['q'].append(record['q'])
        input_data['a'].append(record['a'])
        input_data['candidate_a'].append(record['candidate_a'])
        new_max_qa_len = max(new_max_qa_len, record['qa_len'])

      for i, temp_h_q in enumerate(input_data['h_q']):
        input_data['h_q'][i] = temp_h_q[0:new_max_qa_len]
      for i, temp_h_a in enumerate(input_data['h_a']):
        input_data['h_a'][i] = temp_h_a[0:new_max_qa_len]
      input_data['vgg'] = np.asarray(input_data['vgg'])
      input_data['c3d'] = np.asarray(input_data['c3d'])
      self._batch_queue.put(input_data)

  def next_batch(self):
    """Return a Batch from the batch queue.
    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.
    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._record_queues[0].qsize())

    batch = self._batch_queue.get() # get the next Batch
    return batch