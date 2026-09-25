from __future__ import annotations

import gc, math, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import pydicom
import cv2
from sklearn.metrics import roc_auc_score, f1_score

print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
