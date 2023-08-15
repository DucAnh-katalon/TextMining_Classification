import numpy as np 
import torch

def get_cfg():
    return {
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'EPOCHS' : 6,
    'N_SPLITS' : 5,
    'model_name': "phobert-v2",
    'NUM_EPOCH': 50,
    'warm_up': 5,  
    'max_len':120,
    'batch_size': 64,
    'n_classes':3,
    'output': 'outputs/',
    }




def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True