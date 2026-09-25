# ============================================================
# v4: Imports (lateral 推理: v5 DINOv2 成员 + rad R50 成员)
# ============================================================
from __future__ import annotations
import gc, math, os, re, sys, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision
import pydicom
import cv2
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

IS_MAIN = True
print('Imports OK.')
