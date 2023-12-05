import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())


import pandas as pd

print('Reading CSV..\n')
sys.stdout.flush()

df = pd.read_csv('/mnt/server-home/TUE/20210962/csv/vqa_df.csv')
df['Image_Id'] = df['Image_Id'].apply(lambda x : str(x).zfill(12))

print('Done Reading..\n')
sys.stdout.flush()

# First we need to basically see if we can fetch the images
import torch
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

print('Reading CSV..\n')
sys.stdout.flush()
stacker_embeds = pd.read_csv('/mnt/server-home/TUE/20210962/csv/albef_vqa_embeds.csv')

print('Done Reading..\n')
sys.stdout.flush()

#Prototype Model
from autogluon.tabular import TabularPredictor
time_limit = 300*60  # train various models for ~2 min

predictor = TabularPredictor(label='Predicted_Answers', problem_type='binary').fit(stacker_embeds,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit = time_limit, num_gpus=1)

sys.stdout.flush()
