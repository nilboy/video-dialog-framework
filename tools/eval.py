from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
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
  config['net']['phase'] = 'infer'
  test_data_loader = BasicDataLoader(config['train_data_loader'])
  net = MNet(config['net'])
  solver = Solver(net)
  solver.evaluate(test_data_loader, config['eval_params'])
