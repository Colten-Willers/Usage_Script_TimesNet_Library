import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
import models
from data_provider.data_loader import Dataset_Custom

from models.TimesNet import Model

from tqdm import tqdm

# For preprocessing.

def pre_process(path, target_column, target_range):
    df = pd.read_csv(path)

    min_ = df[target_column].min()
    max_ = df[target_column].max()
    value_range = max_ - min_

    # Other dataset's range
    target_min, target_max = target_range

    # Min-Max scaling formula
    def min_max_scale(value, data_min, data_max, target_min, target_max):
        return ((value - data_min) / (data_max - data_min)) * (target_max - target_min) + target_min
    
    for i, row in enumerate(df[target_column]):
        df[target_column][i] = min_max_scale(row, min_, max_, target_min, target_max)
    
    return df

# For plotting.

def data_provider_custom(test_dataset):
    Data = Dataset_Custom

    data_set = Data(
        root_path="./",
        data_path=test_dataset,
        flag="test",
        size=[96, 48, 96], #args.seq_len, args.label_len, args.pred_len
        )

    data_loader = DataLoader(
        data_set,
        batch_size=32, #25 #batch_size
        shuffle=False, # For testing/usage.
        num_workers=6,
        drop_last=True
        )
    
    return data_set, data_loader

def predict(model, data_loader):
    x_enc_list = []
    predictions_list = []

    # Iterate through batches in the data loader
    for batch in tqdm(data_loader):
        x_enc, x_dec, x_mark_enc, x_mark_dec = batch

        x_enc = x_enc.float()
        x_mark_enc = x_mark_enc.float()
        x_dec = x_dec.float()
        x_mark_dec = x_mark_dec.float()

        x_enc_list.append(x_enc)

        with torch.no_grad():
            predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            predictions_list.append(predictions)
    
    return predictions_list, x_enc_list

def plotting(predictions_list, x_enc_list, amount):
    sequences = []

    for pred in range(len(predictions_list)):
        for sequence_index in range(32):
            sequence = predictions_list[pred][sequence_index]
            sequences.append(sequence)
            for i, element in enumerate(sequences):
                sequences[i] = torch.squeeze(element)

    historical_sequences = []

    for x_enc_ in range(len(x_enc_list)):
        for historical_sequence_index in range(32):
            historical_sequence = x_enc_list[x_enc_][historical_sequence_index]
            historical_sequences.append(historical_sequence)
            for i, element in enumerate(historical_sequences):
                historical_sequences[i] = torch.squeeze(element)



    seqs = [] # The entire thing as one long list of values.

    for s in sequences:
        for val in s:
            seqs.append(val)


    hist = [] # The entire thing as one long list of values.

    for h in historical_sequences:
        for val in h:
            hist.append(val)

    # amount = 4 # In days

    seqs_l = seqs[0:(amount * 96)]
    # hist_l = hist[0:(amount * 96)]
    hist_l = hist[(len(hist) - amount * 96):] # Represents the end, before the forecasting starts.

    plt.figure(figsize=(10, 6))
    plt.plot(torch.tensor([x for x in range(amount * 96)]).reshape(amount * 96,), hist_l, label=f'Example: Historical Data')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Historical Data (x_enc)')
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))

    # plt.plot(torch.tensor([x for x in range(96 * amount)]).reshape(96 * amount,), concated, label=f'Example: Predicted Data')
    plt.plot(torch.tensor([x for x in range(amount * 96)]).reshape(amount * 96,), seqs_l, label=f'Example: Predicted Data')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Predicted data')
    plt.legend()
    plt.grid(True)
    plt.show()