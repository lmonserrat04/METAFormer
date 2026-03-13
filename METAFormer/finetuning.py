from tqdm import tqdm
from logger import Logger
import torch
import copy
import numpy as np

from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn


def train(model,cfg, train_loader, val_loader, criterion, optimizer,device, epochs, patience, scheduler=None):

    early_stopping = True if patience else False
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    best_acc = 0
    counter = 0
    
    log_finetuning = Logger("finetuning_log.txt")

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            epoch_losses = []
            running_loss = 0.0
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                aal, cc200, do160 = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(aal, cc200, do160)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_losses.append(loss.item())
            
            losses.extend(epoch_losses)
            epoch_losses = []



            #Validation training
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                y_true, y_pred = [], []
                for inputs, labels in val_loader:
                    aal, cc200, do160 = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)
                    labels = labels.to(device)

                    outputs = model(aal, cc200, do160)
                    val_loss = criterion(outputs, labels)
                    val_running_loss += val_loss.item()
                    val_losses.append(val_loss.item())

                    y_true.extend(np.argmax(labels.detach().cpu().numpy(), axis=1))
                    yp = F.softmax(outputs, dim=1).detach().cpu().numpy()
                    y_pred.extend(np.argmax(np.where(yp > 0.5, 1, 0), axis=1))

                avg_val_loss = val_running_loss / len(val_loader)
                train_loss = running_loss/len(train_loader)
                acc = accuracy_score(y_true, y_pred)

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                if avg_val_loss < best_val_loss and (best_val_loss - avg_val_loss) >= 1e-3:
                    best_val_loss = avg_val_loss
                    best_model = copy.deepcopy(model)
                    counter = 0
                else:
                    counter += 1
                    if early_stopping and counter >= patience:
                        print(f"Early stopping!, pretrain val loss:{avg_val_loss}")
                        break
                
            
            #tepoch.set_postfix(train_loss=f"{running_loss/len(train_loader)}")
            tepoch.set_postfix(val_loss=f"{avg_val_loss:.4f}")
            log_finetuning.logs(f"fine tuning", epoch, train_loss, avg_val_loss,cfg)

   

   
    return best_model, best_val_loss
    
