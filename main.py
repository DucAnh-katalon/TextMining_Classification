import os
from pathlib import Path 

import pandas as pd
from transformers import PhobertTokenizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import py_vncorenlp

from utils import seed_everything, get_cfg
from CustomDataset import SentimentDataset
from model import SentimentClassifier
from ops import train_model


import wandb
wandb_api = "86b9a5c5a2b9ad64302c105c8653d9a58e7552fc"
wandb.login(key=wandb_api)

def prepare_loaders(df, fold, tokenizer,cfg):
    max_len = cfg.get("max_len")
    batch_size = cfg.get("batch_size")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = SentimentDataset(df_train, tokenizer, max_len=max_len)
    valid_dataset = SentimentDataset(df_valid, tokenizer, max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    dataloaders_dict = {
       'train': train_loader,
        'val': valid_loader

    }
    return dataloaders_dict


cfg = get_cfg()
N_SPLITS = cfg.get("N_SPLITS")
seed_everything(25)
pwd = Path(os.getcwd())
save_dir = pwd / 'models'
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
os.chdir(pwd)

data = pd.read_csv('data - data.csv', usecols=range(0, 3))
data['wseg'] = data['comment'].apply(lambda x: ' '.join(rdrsegmenter.word_segment(x)))

skf = StratifiedKFold(n_splits=N_SPLITS)
for fold, (_, val_) in enumerate(skf.split(X=data, y=data.label)):
    data.loc[val_, "kfold"] = fold


tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base-v2")
device = cfg.get("device")
n_classes = cfg.get("n_classes")
for fold in range(skf.n_splits):
    print(f'-----------Fold: {fold+1} ------------------')
    dataloaders_dict = prepare_loaders(data, fold, tokenizer ,cfg = cfg)
    model = SentimentClassifier(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
   
    train_model(model, dataloaders_dict, criterion, optimizer, cfg)