import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import numpy as np
from scipy.spatial import distance

from loss import mask_nll_loss
from seq2seq import *
from dataloading import *


def convert_response(response, batch_size):
    size1 = len(response)
    size2 = batch_size
    np_res = np.zeros((size1, size2), dtype=np.int64)
    np_lengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = response[i].cpu().numpy()
        for j in range(prov.size):
            np_lengths[j] = np_lengths[j] + 1
            if prov.size > 1:
                np_res[i][j] = prov[j]
            else:
                np_res[i][j] = prov
    res = torch.from_numpy(np_res)
    lengths = torch.from_numpy(np_lengths)
    return res, lengths


def convert_target(target, batch_size):
    size1 = len(target)
    size2 = batch_size
    np_res = np.zeros((size1, size2), dtype=np.int64)
    mask = np.zeros((size1, size2), dtype=np.bool_)
    np_lengths = np.zeros(size2, dtype=np.int64)
    for i in range(size1):
        prov = target[i].cpu().numpy()
        for j in range(prov.size):
            np_lengths[j] = np_lengths[j] + 1
            if prov.size > 1:
                np_res[i][j] = prov[j]
            else:
                np_res[i][j] = prov

            if np_res[i][j] > 0:
                mask[i][j] = True
            else:
                mask[i][j] = False

    res = torch.from_numpy(np_res)
    lengths = torch.from_numpy(np_lengths)
    mask = torch.from_numpy(mask)
    max_target_len = torch.max(lengths)  # .detach().numpy()
    return res, mask, max_target_len


def transform_tensor_to_same_shape_as(tensor, shape):
    size1, size2 = shape
    np_new_t = np.zeros((size1, size2), dtype=np.int64)
    np_new_mask = np.zeros((size1, size2), dtype=np.bool_)
    tensor_size1, tensor_size2 = tensor.size()
    for i in range(tensor_size1):
        for j in range(tensor_size2):
            np_new_t[i][j] = tensor[i][j]
            np_new_mask[i][j] = True
    return torch.from_numpy(np_new_t), torch.from_numpy(np_new_mask)
