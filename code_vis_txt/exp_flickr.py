#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining
import os

# In[3]:


flava_model = FlavaForPreTraining.from_pretrained("facebook/flava-full").cuda()
flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")

device = 'cuda'

# ## Prototyping for Text+Image modality : Img2Text Matching Task
#
# Datasets in Focus : 1. Flickr30k 2.SBU Captions

# Dataset 1: Flickr30k

# In[5]:
caption_data = pd.read_csv('/mnt/server-home/TUE/20210962/flickr/captions.csv')

# In[8]:


# Data Check done, all the images are available in the directory -- done

# Prototype code to get image text scores

from PIL import Image


embeddings = []
item_scores = []
PATH = '/mnt/server-home/TUE/20210962/flickr/Images/'
for i in tqdm(range(len(caption_data))):
    try:
        image = caption_data['image'][i]
        text = caption_data['caption'][i]
        img = Image.open(PATH + image)
        inputs = flava_processor(text=[text], images=[img.convert("RGB")], return_tensors="pt", max_length=77, padding=True,
                                 return_codebook_pixels=True, return_image_mask=True).to(device)  # cuda
        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
        outputs = flava_model(**inputs)

        flava_contrastive_scores = outputs.contrastive_logits_per_image.detach().item()
        flava_itm_scores = torch.nn.functional.softmax(outputs.itm_logits)[0][1].detach().item()
        item_scores.append(flava_contrastive_scores)

        #     mm_embedding = torch.cat((outputs.text_embeddings.flatten(), outputs.image_embeddings.flatten()))
        mm_embedding = outputs.multimodal_masked_output.last_hidden_state

        embeddings.append(mm_embedding.cpu().detach().numpy().flatten())  # Flatten the multimodal embedding

        print("FLAVA contrastive image-text match scores:")

        print("image, caption:", flava_contrastive_scores)
    except ValueError as e:
        print('Exceprion Occured')
        sys.stdout.flush()
        continue


dict_ = {'embeddings': embeddings, 'targets': item_scores}

embeddings_df = pd.DataFrame(dict_)
embeddings_col = pd.DataFrame(embeddings_df['embeddings'], columns=['embeddings'])

tensor_embedding = [torch.Tensor(emx) for emx in embeddings_col['embeddings']]  # Converting to tensors to apply pooling

# Max Pooling the embeddings
print('Starting Pooling..\n')
sys.stdout.flush()

from torch import nn

max_pool = nn.MaxPool1d(2, stride=6)
pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]

pooled_embeddings_final = np.array(pool_embedding, dtype=object)

# Calculating the max sequence length after max pooling
len_ = [arr.shape[0] for arr in pooled_embeddings_final]
max_len = max(len_)
print(f'The max len is {max_len}')
print('Done Pooling..\n')
sys.stdout.flush()

train_stacker_df = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])


embedding_dataframe = pd.concat([train_stacker_df[col].apply(pd.Series) for col in train_stacker_df.columns], axis=1,
                                ignore_index=True)

embedding_dataframe['targets'] = embeddings_df['targets']

embedding_dataframe.to_csv('/mnt/server-home/TUE/20210962/csv/exp_flickr_emb_df.csv')

# In[10]:


# Pass it through AutoGluon Tabular for generating models that work on the dataset

from autogluon.tabular import TabularPredictor

# predictor = TabularPredictor(label='Sentiment', problem_type = 'multiclass')
# predictor.fit(final_stacker).cuda()
time_limit = 400 * 60  # train various models for ~2 min

predictor = TabularPredictor(label='targets', problem_type='regression').fit(embedding_dataframe,
                                                                             num_bag_folds=5, num_bag_sets=1,
                                                                             num_stack_levels=1, time_limit=time_limit,
                                                                             num_gpus=1)

# In[ ]:


# In[ ]:

sys.stdout.flush()

# In[ ]:


# In[ ]:





