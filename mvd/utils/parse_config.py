import yaml

def parse_config(config_path):
  config = yaml.load(open(config_path))
  config['net']['style'] = config['style']
  config['eval_params']['style'] = config['style']
  return config