
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from enum import Enum, auto

class ActivationType(str, Enum):
    RELU = "relu"
    LEAKY_RELU = "lrelu"

class NormLayerType(str, Enum):
    BATCH_NORM = "bn"
    INSTANCE_NORM = "in"
    GROUP_NORM = "gn"
    SYNC_BATCH_NORM = "sync_bn"
    BATCH_CHANNEL_NORM = "bcn"

class BlockType(str, Enum):
    RESIDUAL = "res"
    CONV = "conv"
