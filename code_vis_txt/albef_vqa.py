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


class CustomDataset(Dataset):
    def __init__(self, data, questions, answers, transform=None):
        super().__init__()
        self.DIR = '/mnt/server-home/TUE/20210962/VQA/images/scene_img_abstract_v002_binary_train2017/scene_img_abstract_v002_train2017/'
        self.IMG_TYPE = 'abstract_v002_'
        self.CAT = 'train2015_'
        self.EXT = '.png'
        self.data = data.values
        self.questions = questions.values
        self.answers = answers.values
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


img_size = 384
valid_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()])


# defining a custom collate function
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

batch_size = 500

train_dataset = CustomDataset(df['Image_Id'],df['Question'],df['Answer'], transform = valid_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn )

convert_to_PIL = transforms.ToPILImage()

print('Loading Model..\n')
sys.stdout.flush()
class AlbefVQA(nn.Module):
    def __init__(self, device, batch_size):
        super(AlbefVQA, self).__init__()

        self.device = device
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="albef_vqa",
                                                                                         model_type="vqav2",
                                                                                         is_eval=True,
                                                                                         device=self.device)
        self.answer_candidates = ['yes', 'no']
        self.batch_size = batch_size

        self.model_feat, self.vis_processors_feat, self.txt_processors_feat = load_model_and_preprocess(
            name="albef_feature_extractor", model_type="base", is_eval=True, device=self.device)

    def forward(self, image_batch, question_batch, answer_batch):
        ins_answers = []
        mm_features_flat = []

        for batch_idx in tqdm(range(self.batch_size)):
            image_encoding = self.vis_processors["eval"](image_batch[batch_idx]).unsqueeze(0).to(self.device)
            image_encoding_feat = self.vis_processors_feat["eval"](image_batch[batch_idx]).unsqueeze(0).to(self.device)

            candidate = question_batch[batch_idx]

            question_encoding = self.txt_processors["eval"](candidate)
            question_encoding_feat = self.txt_processors_feat["eval"](candidate)

            sample = {"image": image_encoding, "text_input": [question_encoding]}

            # predicting the answer
            instance_answer = self.model.predict_answers(samples=sample, answer_list=self.answer_candidates,
                                                         inference_method="generate")
            ins_answers.append(instance_answer[0])

            sample_feat = {"image": image_encoding_feat, "text_input": [instance_answer[0]]}

            # Extracting Multimodal Features
            features_multimodal = self.model_feat.extract_features(sample_feat)
            mm_features_flat.append(features_multimodal.multimodal_embeds.detach().cpu().numpy().flatten())

        batch_answers = [ins for ins in ins_answers]

        return batch_answers, mm_features_flat

albef_vqa = AlbefVQA(device = 'cuda', batch_size=500)
print('Done Loading..\n')
sys.stdout.flush()


batch_answers =[]
mm_features = []

print('Starting Inference..\n')
sys.stdout.flush()
for idx, batch in tqdm(enumerate(train_loader)):
    try:
        img, question, answer = batch
        question = list(question)
        answer = list(answer)
        convert_to_PIL = transforms.ToPILImage()
        convert_PIL_array = [convert_to_PIL(img_tensor).convert('RGB') for img_tensor in img]
        ans, mm_features_flat = albef_vqa.forward(convert_PIL_array, question, answer)
        batch_answers.append(ans)
        mm_features.append(mm_features_flat)
    except IndexError as e:
        print('Exception Occurred..\n')
        sys.stdout.flush()

final_embeddings = []
for batch_array in mm_features:
    for array in batch_array:
        final_embeddings.append(array)

predicted_answers = []
for list_item in batch_answers:
    for idx in range(len(list_item)):
        predicted_answers.append(list_item[idx])

stacker_dict = {"embeddings": final_embeddings}
stacker_dict_copy = {"embeddings": final_embeddings, "Predicted_Answers": predicted_answers}

stacker_df_copy = pd.DataFrame(stacker_dict_copy)

stacker_df = pd.DataFrame(stacker_dict)

stacker_embeds = pd.concat([stacker_df[col].apply(pd.Series) for col in stacker_df.columns], axis=1, ignore_index=True)
stacker_embeds['Predicted_Answers'] = stacker_df_copy['Predicted_Answers']
stacker_embeds.to_csv('/mnt/server-home/TUE/20210962/csv/albef_vqa_embeds.csv')

df_dummy = df
df_dummy['Predicted_Answers'] = predicted_answers

ans_dummy = pd.get_dummies(df_dummy['Answer'])
pred_ans_dummy = pd.get_dummies(df_dummy['Predicted_Answers'])

df_dummy["Answer"] = ans_dummy['yes']

df_dummy['Predicted_Answers'] = pred_ans_dummy['yes']

#Inferencing F1-Score
from sklearn.metrics import f1_score

f1_score = f1_score(df_dummy['Answer'], df_dummy['Predicted_Answers'])
print(f'The F1 Score is {f1_score} \n')
sys.stdout.flush()

print('Completed Inference..\n')
sys.stdout.flush()
# !pip install autogluon
#Prototype Model
from autogluon.tabular import TabularPredictor
time_limit = 300*60  # train various models for ~2 min

predictor = TabularPredictor(label='Predicted_Answers', problem_type='binary').fit(stacker_embeds,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit = time_limit, num_gpus=1)

sys.stdout.flush()
