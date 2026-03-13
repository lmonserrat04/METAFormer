import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from METAFormer.dataloader import ImputationDataset, MultiAtlas
from METAFormer.models import METAFormer, METAWrapper
from METAFormer.utils import test
from METAFormer.pretrain import pretrain
from METAFormer.finetuning import train
from logger import Logger


cfg = {
    # --- Paper original ---
    "BATCH_SIZE": 64,
    "LR": 1e-4,             # heads y pretrain
    "LR_ENCODERS": 5e-6,    # encoders preentrenados (differential LR)
    "WEIGHT_DECAY": 0.00,
    "DROP": 0.4,
    "AUG": 0.3,
    "GAMMA": 0.9,
    "PATIENCE": 60,
    "EPOCHS": 2000,
    "LOSS": nn.BCEWithLogitsLoss(),
    "MASK_RATIO" : 0.75,
    # --- Arquitectura ---
    "D_MODEL": 256,
    "DIM_FEEDFORWARD": 128,
    "NUM_ENCODER_LAYERS": 2,
    "NUM_HEADS": 4,
    # --- Infra ---
    "DEVICE": "cuda:0",
    "N_SPLITS": 5,
    "NUM_WORKERS": 8,
    # --- Partes --- #
    "FT_ONLY" : False,
}

DL_KWARGS = dict(
    num_workers=cfg["NUM_WORKERS"],
    pin_memory=True,
)


def build_optimizer(model, cfg, differential=False):
    if differential:
        encoder_params, head_params = [], []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        return optim.AdamW([
            {"params": encoder_params, "lr": cfg["LR_ENCODERS"]},
            {"params": head_params,    "lr": cfg["LR"]},
        ], weight_decay=cfg["WEIGHT_DECAY"])

    return optim.AdamW(
        model.parameters(),
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )

def build_scheduler(optimizer, cfg, is_pretrain=False):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["EPOCHS"],
        eta_min=1e-7,
    )

def pretrain_train_cross_validate(args):

    device = cfg["DEVICE"]
    df = pd.read_csv(args.csv)

    kfold = StratifiedKFold(n_splits=cfg["N_SPLITS"], shuffle=True, random_state=42)
    y = df.LABELS
    x = df.drop("LABELS", axis=1)
    accs = []

    table_cols = ["Fold", "Accuracy", "Precision", "Recall",
                  "F1", "AUC", "AP", "FPR", "FNR", "TPR", "TNR"]
    vals = []

    log_test = Logger("test_log.txt", mode='a')

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
        if fold > 0:   # cambiar a `pass` para correr todos los folds
            break

        print(80 * "=")
        print(f"Fold {fold}")
        print(80 * "=")

        train_df, val_df = train_test_split(
            df.iloc[train_idx], test_size=0.3, random_state=42)
        test_df = df.iloc[test_idx]

       
        # --- Pretrain ---
        pt_model = METAWrapper(
            d_model=cfg["D_MODEL"],
            dim_feedforward=cfg["DIM_FEEDFORWARD"],
            num_encoder_layers=cfg["NUM_ENCODER_LAYERS"],
            num_heads=cfg["NUM_HEADS"],
            dropout=cfg["DROP"],
        ).to(device)

        pt_optim  = build_optimizer(pt_model, cfg, differential=False) 
        pt_scheduler = build_scheduler(pt_optim, cfg, is_pretrain=True)

        pretrained = pt_model

        if not cfg["FT_ONLY"]:

                # --- DataLoaders pretrain ---
            pretrain_loader = DataLoader(ImputationDataset(train_df, cfg["MASK_RATIO"]),
                                        batch_size=cfg["BATCH_SIZE"], shuffle=True,  **DL_KWARGS)
            preval_loader   = DataLoader(ImputationDataset(val_df,cfg["MASK_RATIO"]),
                                        batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)


            pretrained, pt_val_loss = pretrain(
                model=pt_model,
                cfg=cfg,
                train_loader=pretrain_loader,
                val_loader=preval_loader,
                optimizer=pt_optim,
                epochs=cfg["EPOCHS"],
                device=device,
                stage=f"pretrain fold {fold}",
                patience=cfg["PATIENCE"],
                scheduler=pt_scheduler,
            )
            checkpoint = {
                    'model_state_dict': pretrained.state_dict(),
                    'val_loss': pt_val_loss,
            }

            torch.save(checkpoint, f"pretrained_fold{fold}.pth")
            print("Pretraining finished...")
        
        else:
            checkpoint = torch.load(f"pretrained_fold{fold}.pth")
            pretrained.load_state_dict(checkpoint['model_state_dict'], strict=False)
            pt_val_loss = checkpoint['val_loss']
        


        # --- Fine-tuning ---
        model = METAFormer(
            d_model=cfg["D_MODEL"],
            dim_feedforward=cfg["DIM_FEEDFORWARD"],
            num_encoder_layers=cfg["NUM_ENCODER_LAYERS"],
            num_heads=cfg["NUM_HEADS"],
            dropout=cfg["DROP"],
        ).to(device)

        model.aal_encoder.load_state_dict(pretrained.aal_encoder.state_dict())
        model.cc200_encoder.load_state_dict(pretrained.cc200_encoder.state_dict())
        model.dos160_encoder.load_state_dict(pretrained.dos160_encoder.state_dict())

        criterion = cfg["LOSS"].to(device)
        optimizer = build_optimizer(model,    cfg, differential=True)
        scheduler = build_scheduler(optimizer, cfg)

         # --- DataLoaders fine-tuning ---
        train_loader = DataLoader(MultiAtlas(train_df, augment=cfg["AUG"]),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=True,  **DL_KWARGS)
        val_loader   = DataLoader(MultiAtlas(val_df,   augment=0.0),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)
        test_loader  = DataLoader(MultiAtlas(test_df,  augment=0.0),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)

        trained_model, t_val_loss = train(
            model, cfg, train_loader, val_loader, criterion,
            optimizer, device,
            epochs=cfg["EPOCHS"],
            patience=cfg["PATIENCE"],
            scheduler=scheduler, 
        )

        # --- Test ---
        print("Testing...")
        trues, preds, probs = test(trained_model, test_loader, device)

        acc       = accuracy_score(trues, preds)
        accs.append(acc)
        cm        = confusion_matrix(trues, preds)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall    = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1        = 2 * (precision * recall) / (precision + recall)
        auc       = roc_auc_score(trues, probs)
        ap        = average_precision_score(trues, probs)
        fpr       = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        fnr       = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        tpr       = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        tnr       = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        print(f"Fold {fold}: Accuracy: {acc:.4f}")
        print(f"Fold {fold}: Confusion matrix:\n{cm}")

        log_test.log_test(cfg, fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr)
        vals.append([fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr])

    results = pd.DataFrame(vals, columns=table_cols)
    results.to_csv("results.csv", index=False)
    print(results)
    print(80 * "=")
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Std  accuracy: {std_acc:.4f}")

    log_test.log_summary(pt_val_loss,t_val_loss,mean_acc, std_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    pretrain_train_cross_validate(args)