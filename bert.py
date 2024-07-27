import subprocess

# Use pip to install transformers
subprocess.call(['pip', 'install', 'transformers'])

import transformers
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm

# inspired by: https://www.intodeeplearning.com/bert-multiclass-text-classification/


def get_data():
    data = pd.read_table('./data/train.txt', names = ['label', 'review'])
    names = data['label'].unique()
    # Define a mapping from string labels to integer values
    label_mapping = {
        'TRUTHFULPOSITIVE': 0,
        'TRUTHFULNEGATIVE': 1,
        'DECEPTIVEPOSITIVE': 2,
        'DECEPTIVENEGATIVE': 3
    }

    # Use the map function to replace string labels with integer values
    data['label'] = data['label'].map(label_mapping)
    data.groupby( by='label').count()

    return data, names

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=128, padding='max_length',
                            return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

df, label_names = get_data()

train_df, val_df = train_test_split(df, test_size=0.1, random_state=0)
train_df, test_df = train_test_split(train_df, test_size=0.01, random_state=0)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

train_input_ids, train_att_masks = encode(train_df['review'].values.tolist())
val_input_ids, val_att_masks = encode(val_df['review'].values.tolist())
test_input_ids, test_att_masks = encode(test_df['review'].values.tolist())

train_y = torch.LongTensor(train_df['label'].values.tolist())
val_y = torch.LongTensor(val_df['label'].values.tolist())
test_y = torch.LongTensor(test_df['label'].values.tolist())

BATCH_SIZE = 16
train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

val_dataset = TensorDataset(val_input_ids, val_att_masks, val_y)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

n_labels = len(train_df.label.unique())
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=n_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
LEARNING_RATE = 2e-6

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, 
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * EPOCHS)

train_loss_per_epoch = []
val_loss_per_epoch = []


for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)
    '''
    Training
    '''
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
        
        loss = output.loss
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss_per_epoch.append(train_loss / (step_num + 1))              


    '''
    Validation
    '''
    model.eval()
    valid_loss = 0
    valid_pred = []
    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(val_dataloader,desc='Validation')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

            loss = output.loss
            valid_loss += loss.item()
   
            valid_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
        
    val_loss_per_epoch.append(valid_loss / (step_num_e + 1))
    valid_pred = np.concatenate(valid_pred)

    '''
    Loss message
    '''      
    print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_df) / BATCH_SIZE), train_loss / (step_num + 1)))
    print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(val_df) / BATCH_SIZE), valid_loss / (step_num_e + 1)))

print('classifiation report')
print(classification_report(valid_pred, val_df['label'].to_numpy(), target_names=label_names))

model.eval()
test_pred = []
test_loss= 0
with torch.no_grad():
    for step_num, batch_data in tqdm(enumerate(test_dataloader)):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

        loss = output.loss
        test_loss += loss.item()
   
        test_pred.append(np.argmax(output.logits.cpu().detach().numpy(),axis=-1))
test_pred = np.concatenate(test_pred)

print('classifiation report')
print(classification_report(test_pred, test_df['label'].to_numpy(),target_names=label_names))