from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import sys
import tensorflow as tf
import numpy as np
import os
from mvd import metrix

class Solver(object):
  def __init__(self, net):
    self.net = net
    self.net.build_model()

  def train(self, train_data_loader, val_data_loader, test_data_loader, train_params, eval_params):
    np.random.seed(train_params['seed'])
    tf.set_random_seed(train_params['seed'])
    #parse params
    self.epoch_num = train_params['epoch_num']
    self.init_lr = train_params['init_lr']
    self.max_grad_norm = train_params['max_grad_norm']
    self.train_dir = train_params['train_dir']
    self.reload_model = train_params['reload']
    self.lr_decay_step = train_params['lr_decay_step']
    self.lr_decay_rate = train_params['lr_decay_rate']
    self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    self.learning_rate = tf.train.exponential_decay(self.init_lr, self.global_step,
                                           self.lr_decay_step, self.lr_decay_rate, staircase=True)
    tf.summary.scalar('lr', self.learning_rate)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    if train_params.has_key('optimizer'):
      optimizer = eval(train_params['optimizer'])(self.learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.net.loss, tvars), self.max_grad_norm)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    #init ops
    init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    #
    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    ckpt_path = tf.train.latest_checkpoint(self.train_dir)
    if ckpt_path:
      saver.restore(sess, ckpt_path)
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

    for current_epoch_num in range(self.epoch_num):
      train_data_loader.reset()
      for it in range(train_data_loader.batch_num_per_epoch):
        t_begin = time.time()
        input_data = train_data_loader.next_batch()
        feeds = {}
        for key, value in input_data.items():
          feeds[self.net.input_data[key]] = value
        #nilboy = sess.run(self.net.nilboy, feed_dict=feeds)

        loss, _ = sess.run([self.net.loss, self.train_op], feed_dict=feeds)
        t_end = time.time()
        print('Epoch %d step %d, loss = %.2f (%.3f sec/batch)' % ((current_epoch_num, it, loss, t_end-t_begin)))
        sys.stdout.flush()
        if it % 10 == 0:
          summary_str = sess.run(summary_op, feed_dict=feeds)
          summary_writer.add_summary(summary_str, it + current_epoch_num * train_data_loader.batch_num_per_epoch)
        if (it + current_epoch_num * train_data_loader.batch_num_per_epoch) % 1000 == 0 and it != 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=it + current_epoch_num * train_data_loader.batch_num_per_epoch)
          if eval_params['style'] == 'MC':
            self.evaluate_metrix(val_data_loader, eval_params, it + current_epoch_num * train_data_loader.batch_num_per_epoch)
            self.evaluate_metrix(test_data_loader, eval_params, it + current_epoch_num * train_data_loader.batch_num_per_epoch)
          else:
            self.evaluate_loss(val_data_loader, eval_params, it + current_epoch_num * train_data_loader.batch_num_per_epoch)
    sess.close()

  def evaluate(self, data_loader, eval_params, current_iter_num=0):
    """test evaluation
    """
    if eval_params['style'] == 'MC':
      self.evaluate_metrix(data_loader, eval_params, current_iter_num)
    else:
      #
      #
      init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer())
      saver = tf.train.Saver()
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      sess.run(init_op)
      saver.restore(sess, tf.train.latest_checkpoint(eval_params['model_dir']))
      data_loader.reset()
      #
      output_dir = os.path.join(eval_params['model_dir'], 'test')
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      ref_f = open(os.path.join(output_dir, 'ref.txt'), 'a')
      dec_f = open(os.path.join(output_dir, 'dec.txt'), 'a')
      for it in range(data_loader.batch_num_per_epoch):
        input_data = data_loader.next_batch()
        feeds = {}
        for key, value in input_data.items():
          feeds[self.net.input_data[key]] = value
        dec_answer = sess.run(self.net.output_data['answer'], feed_dict=feeds)
        ref_answer = '\n'.join(input_data['a'])

        dec_answer = [_[:_.find('</s>')] for _ in dec_answer]
        dec_answer = '\n'.join(dec_answer)
        dec_f.write(dec_answer)
        ref_f.write(ref_answer)
        ref_f.flush()
        dec_f.flush()
      ref_f.close()
      dec_f.close()
      sess.close()  

  def evaluate_metrix(self, data_loader, eval_params, current_iter_num=0):
    #
    init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer())
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint(eval_params['model_dir']))

    data_loader.reset()
    score = {'Recall@1': 0.0, 'Recall@5': 0.0, 'Recall@10': 0.0, 'MRR': 0.0, 'Mean': 0.0}
    for it in range(data_loader.batch_num_per_epoch):
      input_data = data_loader.next_batch()
      feeds = {}
      for key, value in input_data.items():
        feeds[self.net.input_data[key]] = value
      logits = sess.run(self.net.output_data, feed_dict=feeds)
      rank = metrix.get_rank(logits)
      score['Recall@1'] += metrix.m_recall(1, rank)
      score['Recall@5'] += metrix.m_recall(5, rank)
      score['Recall@10'] += metrix.m_recall(10, rank)
      score['MRR'] += metrix.m_mrr(rank)
      score['Mean'] += metrix.m_rank(rank)
    f = open(os.path.join(eval_params['model_dir'], data_loader.mode + '.log'), 'a')
    f.write('iters %d: ' % current_iter_num)
    for key, value in score.items():
      score[key] = value / data_loader.batch_num_per_epoch
      output_str = '%s\t%.4f\t' % (key, score[key])
      f.write(output_str)
    f.write('\n')
    f.close()
    sess.close()

  def evaluate_loss(self, data_loader, eval_params, current_iter_num=0):
    #
    init_op = tf.group(tf.tables_initializer(), tf.global_variables_initializer())
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)
    saver.restore(sess, tf.train.latest_checkpoint(eval_params['model_dir']))

    data_loader.reset()
    total_loss = 0
    for it in range(data_loader.batch_num_per_epoch):
      input_data = data_loader.next_batch()
      feeds = {}
      for key, value in input_data.items():
        feeds[self.net.input_data[key]] = value
      loss = sess.run(self.net.output_data['loss'], feed_dict=feeds)
      total_loss += loss
    f = open(os.path.join(eval_params['model_dir'], data_loader.mode + '.log'), 'a')
    f.write('iters %d: ' % current_iter_num)
    avg_loss = total_loss / data_loader.batch_num_per_epoch
    output_str = '%.4f' % avg_loss
    f.write(output_str)
    f.write('\n')
    f.close()
    sess.close()
