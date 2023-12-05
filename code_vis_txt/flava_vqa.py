import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())
from torchmultimodal.models.flava.model import flava_model_for_classification
import torch
from torchvision import transforms
from collections import defaultdict
from transformers import BertTokenizer
from functools import partial
import numpy as np
import torch
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
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer,  Data2VecTextModel
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch import nn
from transformers import FlavaProcessor, FlavaForPreTraining
from transformers import AutoProcessor, FlavaModel

print('Reading CSV..\n')
sys.stdout.flush()

df = pd.read_csv('/mnt/server-home/TUE/20210962/csv/vqa_df_sampled.csv')
df['Image_Id'] = df['Image_Id'].apply(lambda x : str(x).zfill(12))
answer_list = df['Answer'].tolist()

new_ans_list = []
for ans in answer_list:
    if ans == 'yes':
        new_ans_list.append(1)
    else:
        new_ans_list.append(0)

df['Answer'] = new_ans_list
print('Done Reading..\n')
sys.stdout.flush()


class CustomDataset(Dataset):
    def __init__(self, data, questions, answers, transform=None):
        super().__init__()
        self.DIR = '/mnt/server-home/TUE/20210962/VQA/images/scene_img_abstract_v002_binary_train2017/scene_img_abstract_v002_train2017/'
        self.IMG_TYPE = 'abstract_v002_'
        self.CAT = 'train2015_'
        self.EXT = '.png'
        self.data = data.values
        self.questions = questions.values.tolist()
        self.answers = answers.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        question = self.questions[index]
        answer = self.answers[index]

        img_path = self.DIR + self.IMG_TYPE + self.CAT + str(img_name) + self.EXT

        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, question, answer


img_size = 224
valid_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()])


# defining a custom collate function
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


batch_size = 10

train_dataset = CustomDataset(df['Image_Id'], df['Question'], df['Answer'],
                              transform=valid_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

max_length = 512
batch_loss = []
predictions = []
batch_embeddings = []
iter_embeddings = []
mm_embeddings = []
prediction_list = []

flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full")
flava_model = FlavaForPreTraining.from_pretrained("facebook/flava-full").cuda()

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
answer_candidates = [1, 0]

flava_class_head = flava_model_for_classification(num_classes=len(answer_candidates))

argmax_func = lambda x: np.argmax(x)

for idx, batch in tqdm(enumerate(train_loader)):
    try:
        torch.cuda.empty_cache()
        img, question, answer = batch
        question = list(question)

        converted_batch_tensors = []
        convert_to_PIL = transforms.ToPILImage()

        img_tensor_list = []
        for idx in tqdm(range(batch_size)):
            img_tensor = img[idx][:3].to('cuda')
            img_tensor_list.append(img_tensor)

        img_tensor_final = torch.stack((img_tensor_list))

        img_tensor_final = F.interpolate(img_tensor_final, size=224)  # model requires dimension to be 224 x 224

        img_tensor_final = img_tensor_final

        print(img_tensor_final.size())

        print('Starting VQA..\n')

        tokenized_question = bert_tokenizer(question)

        # Calculating the max sequence length after max pooling
        len_ = [len(arr) for arr in tokenized_question['input_ids']]
        max_len = max(len_)

        padded_token_list = [list_element + [0] * max(0, max_len - len(list_element)) for list_element in
                             tokenized_question['input_ids']]

        question_tensor = torch.tensor(padded_token_list)

        vqa_outputs = flava_class_head(text=question_tensor, image=img_tensor_final.cpu(), labels=torch.as_tensor(answer))

        # collecting the batch loss
        batch_loss.append(vqa_outputs.loss.item())

        # collecting logits and the predictions ('yes' or 'no' : 'yes = 1', 'no = 0')
        predicted_logits = vqa_outputs.logits.detach().cpu().numpy()
        predicted_labels = [argmax_func(itm) for itm in predicted_logits]
        prediction_list.append(predicted_labels)

        # Collecting the embeddings from the feature processor
        print('Extracting Embeddings..\n')
        sys.stdout.flush()
        torch.cuda.empty_cache()

        # converting tensor to images
        tens_to_img = [convert_to_PIL(item) for item in img_tensor_final]

        inputs = flava_processor(text=question, images=tens_to_img, return_tensors="pt", max_length=77, padding=True,
                                 return_codebook_pixels=True, return_image_mask=True)
        # cuda
        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
        inputs = inputs.to('cuda')
        outputs = flava_model(**inputs)
        mm_embedding = outputs.multimodal_masked_output.last_hidden_state.detach().cpu()
        mm_embeddings.append(mm_embedding)
        torch.cuda.empty_cache()


        print('Done Extraction..\n')
        sys.stdout.flush()
    except IndexError as e:
        continue

# collect all the predictions
final_predictions = []
for arr in prediction_list:
    for pred_label in arr:
        final_predictions.append(pred_label)

dummy_df = df

len(mm_embeddings)
flat_tensors = []
for batch_embeds in mm_embeddings:
    for inst_embeds in batch_embeds:
        flat_tensors.append(inst_embeds.flatten())

#Pooling the embeddings

max_pool = nn.MaxPool1d(4, stride = 8)
pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in flat_tensors]

# flat_embed_list = [item.numpy().tolist() for item in pool_embedding]

pooled_embeddings_final = np.array(pool_embedding, dtype=object)


# Calculating the max sequence length after max pooling
len_ = [arr.shape[0] for arr in pooled_embeddings_final]
max_len = max(len_)
print(f'The max len is {max_len}')

pooled_embeddings_final = pooled_embeddings_final.tolist()

dict_1 = {"embeddings": pooled_embeddings_final}
dict_2 = {"embeddings": pooled_embeddings_final, "targets": final_predictions }

gluon_df_1 = pd.DataFrame(dict_1)
gluon_df_1 = pd.concat([gluon_df_1[col].apply(pd.Series) for col in gluon_df_1.columns], axis=1, ignore_index=True)

gluon_df_2 = pd.DataFrame(dict_2)

gluon_df_1['targets'] = gluon_df_2['targets']

#Don't forget to save gluon_df.
print('Saving Df..\n')
sys.stdout.flush()
gluon_df_1.to_csv('/mnt/server-home/TUE/20210962/csv/flava_vqa_embeds.csv')
print('Done Saving//\n')
sys.stdout.flush()
from autogluon.tabular import TabularPredictor
time_limit = 200*60  # train various models for ~2 min

predictor = TabularPredictor(label='targets', problem_type='binary').fit(gluon_df_1,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit = time_limit, num_gpus=1)

sys.stdout.flush()