from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time

import torch

from .sequence_dataset import SequenceDataset
from .kgc_sequence_dataset import KGCSequenceDataset
from .triple_dataset import TripleDataset
from .kgc_triple_dataset import KGCTripleDataset
from .user_sequential_dataset import UserSequentialDataset

