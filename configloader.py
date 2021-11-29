import yaml
import json

from easydict import EasyDict

TRAIN_CONFIG_FILE = './configurations/train.yml'

train_config = EasyDict(yaml.load(open(TRAIN_CONFIG_FILE).read(), Loader=yaml.Loader))
