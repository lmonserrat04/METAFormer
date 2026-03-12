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
from METAFormer.utils import pretrain, train, test
from logger import Logger


cfg = {
    "BATCH_SIZE": 64,
    "LR": 1e-5,
    "VAL_AFTER": 1,
    "LOSS": nn.CrossEntropyLoss(),
    "WEIGHT_DECAY": 1e-3,
    "DROP": 0.3,
    "AUG": 0.3,
    "GAMMA": 0.995,
    "DEVICE": "cuda:0",
    "PATIENCE": 100,
    "EPOCHS": 2000,
    "N_SPLITS": 2,
    "NUM_WORKERS": 8,
    "FREEZE_EPOCHS": 50,
    "LR_PHASE2": 1e-6,
}

DL_KWARGS = dict(
    num_workers=cfg["NUM_WORKERS"],
    pin_memory=True,
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

    # Logger de tests en modo 'a' — acumula resultados de todos los folds
    # y no sobreescribe entre runs
    log_test = Logger("test_log.txt", mode='a')

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
        if fold == 1:
            continue
        print(80 * "=")
        print(f"Fold {fold}")
        print(80 * "=")

        train_df, val_df = train_test_split(
            df.iloc[train_idx], test_size=0.3, random_state=fold)
        test_df = df.iloc[test_idx]

        # --- DataLoaders ---
        train_loader = DataLoader(MultiAtlas(train_df, augment=cfg["AUG"]),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=True,  **DL_KWARGS)
        val_loader   = DataLoader(MultiAtlas(val_df,   augment=0.0),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)
        test_loader  = DataLoader(MultiAtlas(test_df,  augment=0.0),
                                  batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)

        pretrain_loader = DataLoader(ImputationDataset(train_df),
                                     batch_size=cfg["BATCH_SIZE"], shuffle=True,  **DL_KWARGS)
        preval_loader   = DataLoader(ImputationDataset(val_df),
                                     batch_size=cfg["BATCH_SIZE"], shuffle=False, **DL_KWARGS)

        # --- Cargar pesos preentrenados ---
        pt_model = METAWrapper(d_model=256, dim_feedforward=128,
                               num_encoder_layers=2, num_heads=4,
                               dropout=cfg["DROP"]).to(device)
        pt_model.load_state_dict(torch.load("pretrained.pth"))
        pretrained = pt_model

        # pretrained = pretrain(
        #     model=pt_model, cfg=cfg, train_loader=pretrain_loader,
        #     val_loader=preval_loader,
        #     optimizer=optim.AdamW(pt_model.parameters(),
        #         lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]),
        #     epochs=cfg["EPOCHS"], device=device,
        #     stage=f"pretrain fold {fold}",
        #     patience=cfg["PATIENCE"], scheduler=None)

        print("Pretraining finished...")

        # --- Construir METAFormer y cargar encoders preentrenados ---
        model = METAFormer(d_model=256, dim_feedforward=128,
                           num_encoder_layers=2, num_heads=4,
                           dropout=cfg["DROP"]).to(device)

        model.aal_encoder.load_state_dict(pretrained.aal_encoder.state_dict())
        model.cc200_encoder.load_state_dict(pretrained.cc200_encoder.state_dict())
        model.dos160_encoder.load_state_dict(pretrained.dos160_encoder.state_dict())

        # He initialization solo sobre los heads (capas nuevas)
        # Los encoders tienen pesos preentrenados — no se tocan
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and "encoder" not in name:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        criterion = cfg["LOSS"].to(device)

        # ----------------------------------------------------------------
        # FASE 1 — encoders congelados, solo se entrenan los heads
        # ----------------------------------------------------------------
        print("Fase 1: encoders frozen...")
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

        optimizer_p1 = optim.AdamW(
            [p for name, p in model.named_parameters() if "encoder" not in name],
            lr=cfg["LR"],
            weight_decay=cfg["WEIGHT_DECAY"]
        )

        model, _ = train(model, cfg, train_loader, val_loader, criterion,
                         optimizer_p1, device,
                         epochs=cfg["FREEZE_EPOCHS"],
                         patience=cfg["FREEZE_EPOCHS"],  # sin early stopping en fase 1
                         scheduler=None,
                         return_best_acc=True)

        # ----------------------------------------------------------------
        # FASE 2 — todo descongelado, LR mas bajo
        # ----------------------------------------------------------------
        print("Fase 2: full model unfrozen...")
        for param in model.parameters():
            param.requires_grad = True

        optimizer_p2 = optim.AdamW(
            model.parameters(),
            lr=cfg["LR_PHASE2"],
            weight_decay=cfg["WEIGHT_DECAY"]
        )
        scheduler_p2 = optim.lr_scheduler.ExponentialLR(optimizer_p2, gamma=cfg["GAMMA"])

        trained_model, best_a = train(model, cfg, train_loader, val_loader, criterion,
                                      optimizer_p2, device,
                                      epochs=cfg["EPOCHS"],
                                      patience=cfg["PATIENCE"],
                                      scheduler=scheduler_p2,
                                      return_best_acc=True)

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

        # Loguear resultados del test (modo 'a' — acumula entre runs)
        log_test.log_test(fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr)

        vals.append([fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr])

    results = pd.DataFrame(vals, columns=table_cols)
    results.to_csv("results.csv", index=False)
    print(results)
    print(80 * "=")
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Std  accuracy: {std_acc:.4f}")

    log_test.log_summary(mean_acc, std_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    pretrain_train_cross_validate(args)