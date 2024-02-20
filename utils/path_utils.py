import os

def ensure_dir(path):
    try:
      if not os.path.exists(path):
        os.makedirs(path)
    except FileExistsError:
      print("folder {} already exists".format(path))
