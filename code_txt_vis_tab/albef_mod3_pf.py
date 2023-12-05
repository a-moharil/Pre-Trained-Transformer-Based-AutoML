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



test_df = pd.read_csv('/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/train/train.csv')

#test_df = pd.read_csv('/home/ambarish/Desktop/scripts/Image_text_tabular/data/petfinder-adoption-prediction/train/train.csv')
test_df['text'] = test_df['Name'] + ", " + test_df['Description']

test_df = test_df.drop(['Name', 'Description'], axis = 1)

#Fill na values

from collections import Counter
dummy_text = test_df['text'][0]
test_df['text'].fillna(dummy_text,inplace = True)

tabular_df = test_df
tabular_df = tabular_df.drop(['text'], axis = 1)
print("Done Processing Df..\n")

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
        img_path = self.path + img_name + "-" + str(1) + ".jpg"
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
PATH = '/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/train_images/'
train_dataset = CustomDataset(test_df['PetID'], test_df['text'], PATH, transform=valid_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)


from tqdm import tqdm
model, vis_processors, txt_processors = load_model_and_preprocess(name="albef_feature_extractor",
                                model_type="base",  is_eval=True, device=device)

convert_to_PIL = transforms.ToPILImage()
device = 'cuda'
mm_features_flat = []
for idx, batch in tqdm(enumerate(train_loader)):
    #print(batch[batch_idx][modality]) (0: img, 1: caption)
    for batch_idx in range(batch_size):
        try:
            vis_encoding = [vis_processors["eval"](convert_to_PIL(batch[batch_idx][0]).convert("RGB")).unsqueeze(0).to(device)]
            txt_encoding = [txt_processors["eval"](str(batch[batch_idx][1][:511]))]
            sample = {"image": vis_encoding[0], "text_input": txt_encoding[0]}
            features_multimodal = model.extract_features(sample)
            mm_features_flat.append(features_multimodal.multimodal_embeds.detach().cpu().numpy().flatten())
        except IndexError as e:
            continue
sys.stdout.flush()


from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label='AdoptionSpeed')
predictor.fit(tabular_df)

tab_embeds = predictor.extract_embedding(tabular_df)

embedding_ = np.array(tab_embeds)

# Calculating the max sequence length
len_ = [arr.shape[0] for arr in embedding_]
max_len = max(len_)
print(f'The max len is {max_len}')
sys.stdout.flush()


tensor_embedding = [torch.Tensor(emx) for emx in embedding_]

from torch import nn

max_pool = nn.MaxPool1d(4, stride=8)
pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]

pooled_embeddings_final = np.array(pool_embedding, dtype=object)


# Calculating the max sequence length after max pooling
len_ = [arr.shape[0] for arr in pooled_embeddings_final]
max_len = max(len_)
print(f'The max len is {max_len}')
sys.stdout.flush()


# Now we will create a stacking model to stack the embeddings using a shallow stacker.
train_stacker_df = pd.DataFrame(pooled_embeddings_final)
print(train_stacker_df.shape)
sys.stdout.flush()

import pandas as pd

stacker_df = pd.concat([train_stacker_df, tabular_df['AdoptionSpeed']], axis=1)


# Creating a feature dataframe for the obtained embedddings
#embedding_dataframe = pd.concat([train_stacker_df[col].apply(pd.Series) for col in train_stacker_df.columns], axis=1,
 #                               ignore_index=True)

#final_stacker = pd.concat([stacker_df,  tabular_df['AdoptionSpeed']], axis=1)



stacker_df.to_csv('/mnt/server-home/TUE/20210962/csv/albef_mod3_pf_embeds.csv')

from autogluon.tabular import TabularPredictor

# predictor = TabularPredictor(label='Sentiment', problem_type = 'multiclass')
# predictor.fit(final_stacker).cuda()
time_limit = 120 * 60 # train various models for ~2 min

predictor = TabularPredictor(label='AdoptionSpeed', problem_type='multiclass').fit(stacker_df,
                                                                                num_bag_folds=5, num_bag_sets=1,
                                                                                num_stack_levels=1,
                                                                                time_limit=time_limit, num_gpus=1,
                                                                                hyperparameters={
                                                                                    'NN_TORCH': {'num_epochs': 2},
                                                                                    'GBM': {'num_boost_round': 20}},
                                                                                # last  argument is just for quick demo here, omit it in real applications
                                                                                )


sys.stdout.flush()


