#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch


# In[2]:


# Definitions for later use

# concatenates sentences into texts. Works for dataframes that have "########################" as last row, marking the end of a review
def get_text(dataframe):

    text = ''

    # if  no "########################" in the last row, "- 1" should be dropped
    for x in range(len(dataframe) - 1):
        text = text + ' ' + dataframe['text'].iloc[x]

    return text

# transforms sentence-split dataframe into whole reviews
def transform_text(file):

    # Split each review
    split_id = file.index[file['text'] == '###########################'].tolist()
    split_id = [-1] + split_id + [len(file)]
    split_df_test = [file.iloc[start+1:end+1] for start, end in zip(split_id[:-1], split_id[1:])]
    split_df_test = [df for df in split_df_test if not df.empty]
    text_list = []

    labels = file[file['text'] == '###########################'].iloc[:,1:].reset_index(drop=True)

    # get text from each dataframe that constitutes a single review
    for small_df in split_df_test:
        text_list.append(get_text(small_df))

    text_df = pd.DataFrame({'text': text_list}).reset_index(drop=True)

    final_df = pd.concat([text_df, labels], axis=1)

    return final_df

# transforms file to remove whole-review marked rows and leave sentence only rows
def transform_sentences(file):

    sentence_df = file[file['text'] != '###########################']

    return sentence_df

# allows to recreate the dataframe with original columns for whole-review data
def restore_text_dataframe(predictions):

    cols = ['text', 'Joy', 'Trust', 'Anticipation', 'Surprise', 'Fear', 'Sadness',
       'Disgust', 'Anger', 'Positive', 'Negative', 'Neutral']

    review = ['###########################'] * 167

    pred_df = pd.DataFrame(predictions)
    restored_text = pd.concat([pd.Series(review), pred_df], axis=1)
    restored_text.columns = cols

    return restored_text

# allows to recrete the dataframe with original columns for sentence data
def restore_sent_dataframe(predictions, original_df):

    cols = ['text', 'Joy', 'Trust', 'Anticipation', 'Surprise', 'Fear', 'Sadness',
       'Disgust', 'Anger', 'Positive', 'Negative', 'Neutral']

    sentences = original_df[original_df['text'] != '###########################'].reset_index(drop=True)
    pred_df = pd.DataFrame(predictions)

    restored_sent = pd.concat([sentences, pred_df], axis=1)
    restored_sent.columns = cols

    return restored_sent

# combines sentence and text data into single dataframe based on indices from the original dataframe
def combine_data(text_df, sent_df, original_df):

    sentence_indices = original_df.index[original_df['text'] != '###########################'].tolist()
    review_indices = original_df.index[original_df['text'] == '###########################'].tolist()

    sent_df['original_index'] = sentence_indices
    text_df['original_index'] = review_indices

    df_combined = pd.concat([sent_df, text_df], ignore_index=True)
    df_combined = df_combined.sort_values(by='original_index').reset_index(drop=True)

    df_final = df_combined.drop(['original_index','text'], axis=1)

    # returns a boolean dataframe with data for emotions
    df_bool = df_final.astype(bool)

    return df_bool


# In[ ]:


# get raw data for predictions 
def get_test_data(model, test_dataloader, device):

    model.eval()

    labels_test = []
    predictions_test = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.float()

            outputs = model(input_ids, attention_mask=attention_mask)
            
            predictions = outputs.logits
            predictions = torch.sigmoid(predictions)
            
            predictions_r = predictions.cpu().numpy()

            # Append labels and predictions for F1 calculation
            labels_test.extend(labels.cpu().numpy())
            predictions_test.extend(predictions_r)

    return labels_test, predictions_test

def get_test_evaluation(labels_test, preds_test, classes):

    test_dict = {}

    f1_macro_score = f1_score(labels_test, preds_test, average='macro')
    test_dict['f1_macro'] = f1_macro_score

    # Compute metrics for each label
    for i, label in enumerate(classes):
        f1_test = round(f1_score(labels_test[:, i], preds_test[:, i]), 4)
        test_dict[label] = f1_test

    return test_dict

