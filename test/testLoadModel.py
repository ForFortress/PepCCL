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

model_paths = 'PATH.pth'

path_train_data_path = r'/data/Dataset2_train.tsv'
path_test_data_path = r'/data/Dataset2_test.tsv'
t5_path_path = ""
result_file = ""


"""************************************ START ***************************************"""
def load_data(config):
    train_iter_orgin, test_iter = data_loader_t5.load_data(config) # change
    return train_iter_orgin, test_iter


def model_eval(data_iter, model, criterion, config, model_name):
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            input, label = batch
            label = torch.tensor(label, dtype=torch.long).cuda(device)
            lll = label.clone()
            label = torch.unsqueeze(label, 0)
            logits = model.get_logits(input)
            output = model.forward(input)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(lll.cpu().detach().numpy())

            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)

            loss = criterion(logits, label)
            avg_loss += loss.item()

            logits = torch.unsqueeze(logits, 0)
            label = torch.unsqueeze(label, 0)
            pred_prob_all = F.softmax(logits, dim=2)
            pred_prob_positive = pred_prob_all[:, :, 1]
            positive = torch.empty([0], device=device)

            pred_prob_sort = torch.max(pred_prob_all, 2)

            pred_class = pred_prob_sort[1]
            p_class = torch.empty([0], device=device)
            la = torch.empty([0], device=device)
			
            positive = torch.cat([positive, pred_prob_positive[0][:]])
            p_class = torch.cat([p_class, pred_class[0][:]])
            la = torch.cat([la, label[0][:]])

            corre = (pred_class == label).int()
            corrects += corre.sum()

            iter_size += label.size(1)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, la.float()])
            pred_prob = torch.cat([pred_prob, positive])

    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)


    label_real_np = label_real.cpu().numpy()
    pred_prob_np = pred_prob.cpu().numpy()


    fpr, tpr, _ = roc_curve(label_real_np, pred_prob_np)
    roc_auc = auc(fpr, tpr)
        
    precision, recall, _ = precision_recall_curve(label_real_np, pred_prob_np)
    ap = average_precision_score(label_real_np, pred_prob_np)

    Mxx_test = {}
    Mxx_test['precision'] = precision.tolist()
    Mxx_test['recall'] = recall.tolist()
    Mxx_test['fpr'] = fpr.tolist()
    Mxx_test['tpr'] = tpr.tolist()

    Mxx_test['roc_auc'] = roc_auc
    Mxx_test['ap'] = ap
    # to be soon... 
