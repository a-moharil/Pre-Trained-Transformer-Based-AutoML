import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining
import os

from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

PATH = '/mnt/server-home/TUE/20210962/albef/Images/'
device = 'cuda'

print('Reading DF...\n')
sys.stdout.flush()
albef_data = pd.read_csv('/mnt/server-home/TUE/20210962/csv/exp_albef_emb_df.csv', nrows=30000)
albef_data = pd.DataFrame(albef_data)
print('Done Rading...\n')
sys.stdout.flush()
from PIL import Image

#Doing the pooling operations:
print('Starting Pooling...\n')
sys.stdout.flush()

new_df = albef_data.drop(['similarity'], axis=1)
embed_arr = new_df.to_numpy()
embed_tensors = [torch.Tensor(tensor) for tensor in embed_arr]
max_pool = nn.MaxPool1d(3, stride=8)
pool_embeds = [max_pool(tensor.unsqueeze(0)).squeeze(0).numpy() for tensor in embed_tensors]

len_ = [arr.shape[0] for arr in pool_embeds]
max_len = max(len_)
print(f'The max len is {max_len}')
sys.stdout.flush()
train_stacker_df = pd.DataFrame(pool_embeds)
train_stacker_df['similarity'] = albef_data['similarity']
print('Done Pooling...\n')
sys.stdout.flush()




from autogluon.tabular import TabularPredictor
time_limit = 200*60  # train various models for ~2 min

predictor = TabularPredictor(label='similarity', problem_type='regression').fit(train_stacker_df,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit = time_limit, num_gpus=1)

sys.stdout.flush()
