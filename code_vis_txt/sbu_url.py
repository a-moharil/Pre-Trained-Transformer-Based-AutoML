import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, FlavaTextModel
from auto_mm_bench.datasets import dataset_registry
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, FlavaTextModel
from transformers import AutoTokenizer, Data2VecTextModel
from auto_mm_bench.datasets import dataset_registry
from autogluon.tabular import TabularPredictor

#Image Crawling
import requests 
import urllib.request
from urllib.request import urlopen
from tqdm import tqdm

DIR_URL = '/mnt/server-home/TUE/20210962/SBU/dataset/SBU_captioned_photo_dataset_urls.txt'
DIR_IMG = "/mnt/server-home/TUE/20210962/SBU/dataset/images/"

with open(DIR_URL) as f:
    url_list = f.readlines()
    f.close()

for url in tqdm(url_list):
    try:
        image_name = url.split('/')[-1].split('\n')[0]
        url = url.split('\n')[0]
        urlopen(url)
        urllib.request.urlretrieve(url, DIR_IMG + image_name)
    except:
        pass

sys.stdout.flush()

