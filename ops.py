from tqdm import tqdm 
import torch
import torch.nn as nn
import numpy as np
import wandb
import time
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup

def train_model(model, dataloaders, criterion, optimizer, cfg):
    run = wandb.init(project='TextMining', 
                 config=cfg,
                 group='phobert', 
                 job_type='train')
    
    since = time.time()
    device = cfg['device']
    artifact = wandb.Artifact(name=cfg['model_name'], type='model')
    cooldown_epochs = cfg['warm_up']
    num_epochs = cfg['NUM_EPOCH'] + cooldown_epochs
    val_acc_history = []
    num_steps_per_epoch = len(dataloaders['train'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_steps_per_epoch * cooldown_epochs, num_training_steps= num_epochs * num_steps_per_epoch)
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_pbar = tqdm(range(num_epochs))

    for epoch in epoch_pbar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            if (phase == 'val') and ((epoch % 2) !=0):
                continue
                
                
            running_loss = 0.0
            running_corrects = 0
            num_steps_per_epoch = len(dataloaders[phase])
            num_updates = epoch * num_steps_per_epoch
            
            # Iterate over data.
            for idx, (data) in enumerate(dataloaders[phase]):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_masks'].to(device)
                labels = data['targets'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                            )
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # statistics 
                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)                
                   
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            log_epoch = {
                f"{phase}_loss": epoch_loss,
                f"{phase}_acc" : epoch_acc 
            }
            epoch_pbar.set_postfix(log_epoch)
            wandb.log(log_epoch)
            
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{cfg['output']}_best.pt")
            if phase == 'val':
                val_acc_history.append(epoch_acc)  



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(model.state_dict(), f"{cfg['output']}_last.pt")
    artifact.add_file(f"{cfg['output']}_last.pt")
    artifact.add_file(f"{cfg['output']}_best.pt")
    run.log_artifact(artifact)
    run.finish()        
    model.load_state_dict(torch.load(f"{cfg['output']}_best.pt"))  
    return model, val_acc_history
