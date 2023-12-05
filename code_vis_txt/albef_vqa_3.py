import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())

from tqdm import tqdm
import pandas as pd

print('Reading Image Data CSV..\n')
sys.stdout.flush()

df = pd.read_csv('/mnt/server-home/TUE/20210962/csv/vqa_df.csv')
df['Image_Id'] = df['Image_Id'].apply(lambda x : str(x).zfill(12))

print('Done Reading Image Data Csv..\n')
sys.stdout.flush()


for idx in tqdm(range(len(df))):
    if df['Answer'][idx] == 'yes':
        df['Answer'][idx] = 1
    else:
        df['Answer'][idx] = 0




#Stacker embeds load
print('Reading Stacker CSV..\n')
sys.stdout.flush()

stacker_df = pd.read_csv('/mnt/server-home/TUE/20210962/csv/albef_vqa_embeds.csv')
stacker_df['Predicted_Answers'] = df['Answer'][:len(stacker_df)]

print('Done Reading Stacker Csv..\n')
sys.stdout.flush()

#Starting AutoGluon
print('Reading Stacker CSV..\n')
sys.stdout.flush()

from autogluon.tabular import TabularPredictor
time_limit = 300*60  # train various models for ~2 min

predictor = TabularPredictor(label='Predicted_Answers', problem_type='binary').fit(stacker_df,
    num_bag_folds=5, num_bag_sets=1, num_stack_levels=1, time_limit = time_limit, num_gpus=1)

sys.stdout.flush()
