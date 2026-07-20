import sys
import os
import json
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random
from datetime import datetime


from util import data_loader_t5
from configuration import config as cf
from util import util_metric

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import transformers
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import model
import tripletLoss

# GPU
device = torch.device("cuda:0")

path_train_data_path = r'/data/Dataset1_train.tsv'
path_test_data_path = r'/data/Dataset1_test.tsv'
t5_path_path = ""
result_file = ""


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=1)
    time_start = time.time()

    config = load_config()

    torch.cuda.set_device(device)

    train_iter, test_iter = load_data(config)

    # to be soon...
