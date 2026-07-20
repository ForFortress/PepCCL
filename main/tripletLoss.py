import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import seaborn as sns
import random
import torch.nn.functional as F


# GPU
device = torch.device("cuda:0")


def TL(output,label,logits,epoch):
    z_zero = torch.tensor([0.0]).to(device)
    z_one = torch.tensor([1.0]).to(device)

    margin = torch.tensor(0.2, dtype=torch.float32).cuda(device)

    matrix0 = output[label == 0]
    matrix1 = output[label == 1]

    c0_center = torch.mean(matrix0, dim=0)
    c1_center = torch.mean(matrix1, dim=0)
    num0 = matrix0.shape[0]
    num1 = matrix1.shape[0]
    num_executions = (num0 + num1) // 2
    num_exe = num_executions // 2

    indices0 = torch.randint(high=num0, size=(num_exe * 2,))
    indices1 = torch.randint(high=num1, size=(num_exe * 2,))

    class0_center_normalized = F.normalize(c0_center, p=2, dim=0)
    cs0 = torch.nn.functional.cosine_similarity(matrix0, class0_center_normalized.unsqueeze(0), dim=1)
    cs0 = (cs0 + 1) * 0.5

    class1_center_normalized = F.normalize(c1_center, p=2, dim=0)
    cs1 = torch.nn.functional.cosine_similarity(matrix1, class1_center_normalized.unsqueeze(0), dim=1)
    cs1 = (cs1 + 1) * 0.5

    norm0 = torch.norm(matrix0, dim=1, keepdim=True)
    norm1 = torch.norm(matrix1, dim=1, keepdim=True)
    dot_product = torch.matmul(matrix0, matrix1.t())
    cs01 = dot_product / (norm0 * norm1.t())
    cs01 = (cs01 + 1) * 0.5

    dot_product_00 = torch.matmul(matrix0, matrix0.t())
    cs00 = dot_product_00 / (norm0 * norm0.t())
    cs00 = (cs00 + 1) * 0.5

    dot_product_11 = torch.matmul(matrix1, matrix1.t())
    cs11 = dot_product_11 / (norm1 * norm1.t())
    cs11 = (cs11 + 1) * 0.5

    class0_center_normalized = F.normalize(c0_center, p=2, dim=0)
    cs1_to_center0 = torch.nn.functional.cosine_similarity(matrix1, class0_center_normalized.unsqueeze(0), dim=1)
    cs1_to_center0 = (cs1_to_center0 + 1) * 0.5

    class1_center_normalized = F.normalize(c1_center, p=2, dim=0)
    cs0_to_center1 = torch.nn.functional.cosine_similarity(matrix0, class1_center_normalized.unsqueeze(0), dim=1)
    cs0_to_center1 = (cs0_to_center1 + 1) * 0.5

    ################################## LET DANCE #####################################

    pp = cs1[indices1[0:num_exe]]
    nn = cs01[ indices0[0:num_exe]  ,  indices1[0:num_exe]  ]   
    loss1 = torch.max(torch.zeros_like(pp), 1 - pp + nn + margin)  ** 3
    Loss1 = torch.mean(loss1)

    pp = cs0[indices0[num_exe : 2*num_exe]]
    nn = cs01[ indices0[num_exe : 2*num_exe] , indices1[num_exe:2*num_exe] ]   
    loss2 = torch.max(torch.zeros_like(pp), 1 - pp + nn + margin)  ** 2
    Loss2 = torch.mean(loss2)

    Loss = Loss1 * 0.5 + Loss2 * 0.5

    return Loss
