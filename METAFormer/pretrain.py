from tqdm import tqdm
from logger import Logger
from METAFormer.utils import MaskedMSELoss
import torch
import copy



def pretrain(model,cfg, train_loader, val_loader, optimizer,device, epochs, stage, patience, scheduler=None):
    early_stopping = True if patience else False
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    log_pretraining = Logger("pretrain_log.txt")

    crit_aal = MaskedMSELoss()
    crit_cc200 = MaskedMSELoss()
    crit_dos160 = MaskedMSELoss()

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            epoch_losses = []
            running_loss = 0.0
            model.train()
            for i, ((aal, cc200, dos160), (aal_masked, cc200_masked, dos160_masked), (aal_mask, cc200_mask, dos160_mask)) in enumerate(train_loader):
                
                aal, cc200, dos160 = aal.to(device), cc200.to(device), dos160.to(device)
                aal_masked, cc200_masked, dos160_masked = aal_masked.to(device), cc200_masked.to(device), dos160_masked.to(device)
                aal_mask, cc200_mask, dos160_mask = aal_mask.to(device), cc200_mask.to(device), dos160_mask.to(device)
                optimizer.zero_grad()

                outputs = model(aal_masked, cc200_masked, dos160_masked)

                # FIX: pasar las mascaras bool, no los inputs enmascarados
                loss_aal = crit_aal(outputs[0], aal, aal_mask.bool())
                loss_cc200 = crit_cc200(outputs[1], cc200, cc200_mask.bool())
                loss_dos160 = crit_dos160(outputs[2], dos160, dos160_mask.bool())
                
                loss = loss_aal + loss_cc200 + loss_dos160
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_losses.append(loss.item())

            if scheduler:
                scheduler.step()
            losses.extend(epoch_losses)
            epoch_losses = []
            


            #Validation pretraining
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                for j, ((aal, cc200, dos160), (aal_masked, cc200_masked, dos160_masked), (aal_mask, cc200_mask, dos160_mask)) in enumerate(val_loader):
                    aal, cc200, dos160 = aal.to(device), cc200.to(device), dos160.to(device)
                    aal_masked, cc200_masked, dos160_masked = aal_masked.to(device), cc200_masked.to(device), dos160_masked.to(device)
                    aal_mask, cc200_mask, dos160_mask = aal_mask.to(device), cc200_mask.to(device), dos160_mask.to(device)

                    outputs = model(aal_masked, cc200_masked, dos160_masked)

                    # FIX: pasar las mascaras bool, no los inputs enmascarados
                    loss_aal = crit_aal(outputs[0], aal, aal_mask.bool())
                    loss_cc200 = crit_cc200(outputs[1], cc200, cc200_mask.bool())
                    loss_dos160 = crit_dos160(outputs[2], dos160, dos160_mask.bool())
                    loss = loss_aal + loss_cc200 + loss_dos160

                    val_running_loss += loss.item()
                    epoch_losses.append(loss.item())
                    val_losses.append(loss.item())

                avg_val_loss = val_running_loss / len(val_loader)
                train_loss = running_loss/len(train_loader)

                if avg_val_loss < best_val_loss and (best_val_loss - avg_val_loss) >= 1e-3:
                    best_val_loss = avg_val_loss
                    counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    counter += 1
                    if early_stopping and counter >= patience:
                        print(f"Early stopping!, pretrain val loss:{avg_val_loss}")
                        break
                
                #tepoch.set_postfix(train_loss=f"{train_loss:.4f}")
                tepoch.set_postfix(val_loss=f"{avg_val_loss:.4f}")
                #tepoch.set_postfix(counter=f"{counter}")
                log_pretraining.logs(stage, epoch, train_loss, avg_val_loss,cfg)

    
    return best_model, best_val_loss
