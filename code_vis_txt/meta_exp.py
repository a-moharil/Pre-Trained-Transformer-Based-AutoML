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


predictor = TabularPredictor.load('/mnt/server-home/TUE/20210962/AutogluonModels/ag-20230420_122948/')
print(predictor.leaderboard(extra_info=True, silent=True))
print(type(predictor.leaderboard(extra_info=True, silent=True)))
predictor.leaderboard(extra_info=True, silent=True).to_csv('/mnt/server-home/TUE/20210962/csv/albef_itm.csv')
sys.stdout.flush()

predictor = TabularPredictor.load('/mnt/server-home/TUE/20210962/AutogluonModels/ag-20230420_121823/')
print(predictor.leaderboard(extra_info=True, silent=True))
print(type(predictor.leaderboard(extra_info=True, silent=True)))
predictor.leaderboard(extra_info=True, silent=True).to_csv('/mnt/server-home/TUE/20210962/csv/flickr_itm.csv')
sys.stdout.flush()


sys.stdout.flush()
