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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining
import os
from transformers import AutoTokenizer,  Data2VecTextModel
from lavis.models import load_model_and_preprocess



device = 'cuda'
print('Processing Df..\n')
sys.stdout.flush()



cd_18 = pd.read_csv("/mnt/server-home/TUE/20210962/CD18/CD18.csv")

pic_url_list = cd_18['Picture'].values.tolist()
cd_18 = cd_18.reset_index()
cd_18_tabular = cd_18.drop(['Picture'], axis=1)
print("Done Processing Df..\n")
sys.stdout.flush()
###############################################

# from autogluon.multimodal import MultiModalPredictor
#
# predictor = MultiModalPredictor(label='Price')
# predictor.fit(cd_18_tabular)
#
# tab_embeds_op = predictor.extract_embedding(cd_18_tabular)
#
# print(type(tab_embeds_op))
# sys.stdout.flush()
######################################################
class CustomDataset(Dataset):
    def __init__(self, data, captions, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.captions = captions.values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img_name = self.data[index]
        label = self.captions[index]
        img_path = self.path + str(img_name) + ".jpg"
        try:
            image = Image.open(img_path)
            if self.transform is not None:
                image = self.transform(image)
        except FileNotFoundError as e:
            image = Image.open('/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/test.jpg')
            if self.transform is not None:
                image = self.transform(image)

        return image, label

img_size = 224
valid_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()])


# defining a custom collate function
def collate_fn(batch):
    return [(torch.tensor(dp[0]), dp[1]) for dp in batch]


batch_size = 100
PATH = '/mnt/server-home/TUE/20210962/CD18/CD18-Images/'
train_dataset = CustomDataset(cd_18['index'], cd_18['Model'], PATH, transform=valid_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)


from tqdm import tqdm
##FLAVA MODEL

flava_model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")#      

# Getting tabular + text embeddings
convert_to_PIL = transforms.ToPILImage()
device = 'cuda'
mm_features_flat = []
for idx, batch in tqdm(enumerate(train_loader)):
    #print(batch[batch_idx][modality]) (0: img, 1: caption)
    for batch_idx in range(batch_size):
        try:
            inputs = flava_processor(text=str(batch[batch_idx][1][:511]), images=[convert_to_PIL(batch[batch_idx][0]).convert("RGB")], return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True) #cuda
            outputs = flava_model(**inputs)
            mm_embedding = outputs.multimodal_masked_output.last_hidden_state
            mm_features_flat.append(mm_embedding.detach().numpy().flatten()) #Flatten the multimodal embedding
        except IndexError as e:
            continue
sys.stdout.flush()


from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='Price')
predictor.fit(cd_18_tabular)

tab_embeds = predictor.extract_embedding(cd_18_tabular)

tab_embed_df = pd.DataFrame(tab_embeds)

mm_embed_df = pd.DataFrame(mm_features_flat)

embedding_df = pd.concat([tab_embed_df, mm_embed_df.reindex(tab_embed_df.index)], axis=1)
embedding_df = embedding_df.fillna(0)

embedding_array = np.array(embedding_df)

# Calculating the max sequence length
len_ = [arr.shape[0] for arr in embedding_array]
max_len = max(len_)
print(f'The max len is {max_len}')
sys.stdout.flush()

tensor_embedding = [torch.Tensor(emx) for emx in embedding_array]

from torch import nn

max_pool = nn.MaxPool1d(2, stride=4)
pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]

pooled_embeddings_final = np.array(pool_embedding, dtype=object)


# Calculating the max sequence length after max pooling
len_ = [arr.shape[0] for arr in pooled_embeddings_final]
max_len = max(len_)
print(f'The max len is {max_len}')
sys.stdout.flush()

stacker_df = pd.DataFrame(pooled_embeddings_final)
stacker_df = stacker_df.replace(0,np.nan)
stacker_df['Price'] = cd_18['Price'][:len(stacker_df)]
stacker_df.to_csv('/mnt/server-home/TUE/20210962/csv/flava_mod3_cd18_embeds.csv')

from autogluon.tabular import TabularPredictor

# predictor = TabularPredictor(label='Sentiment', problem_type = 'multiclass')
# predictor.fit(final_stacker).cuda()
time_limit = 300 * 60 # train various models for ~2 min

predictor = TabularPredictor(label='Price', problem_type='regression').fit(stacker_df,
                                                                                num_bag_folds=5, num_bag_sets=1,
                                                                                num_stack_levels=1,
                                                                                time_limit=time_limit, num_gpus=1,
                                                                                hyperparameters={
                                                                                    'NN_TORCH': {'num_epochs': 2},
                                                                                    'GBM': {'num_boost_round': 20}},
                                                                                # last  argument is just for quick demo here, omit it in real applications
                                                                                )


sys.stdout.flush()

