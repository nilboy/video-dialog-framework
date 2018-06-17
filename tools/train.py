from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from setproctitle import setproctitle
sys.path.insert(0, './')

import mvd
from mvd.data import *
from mvd.net import *
from mvd.solver import *
from mvd.utils import *

import argparse

if __name__ == '__main__':
  #
  parser = argparse.ArgumentParser()
  parser.add_argument('-c','--conf', help='configure file', type = str, required=True)
  args = parser.parse_args()
  config = parse_config(args.conf)
  setproctitle(config['process_name'])
  train_data_loader = BasicDataLoader(config['train_data_loader'])
  val_data_loader = BasicDataLoader(config['val_data_loader'])
  test_data_loader = BasicDataLoader(config['test_data_loader'])
  net = MNet(config['net'])
  solver = Solver(net)
  solver.train(train_data_loader, val_data_loader, test_data_loader, config['train_params'], config['eval_params'])
