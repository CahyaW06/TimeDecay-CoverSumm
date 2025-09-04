import os, json, numpy as np

def np_encoder(object):
  if isinstance(object, np.generic):
    return object.item()

def dump_data(data, path):
  directory = os.path.dirname(path)
  if not os.path.exists(directory):
    os.makedirs(directory)
    
  with open(path, 'w') as f:
    json.dump(data, f, default=np_encoder, indent=4)