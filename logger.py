import logging
import os


class Logger:

    def __init__(self, log_path, mode='w'):
        self.log_path = log_path

        self.logger = logging.getLogger(log_path)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(log_path, mode=mode)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)

    def logs(self, about, epoch, train_loss, val_loss, cfg=None):
        if epoch == 0 and cfg is not None:
            cfg_msg = " | ".join(f"{k}:{v}" for k, v in cfg.items())
            self.logger.info(f"[config] {cfg_msg}")

        msg = f"[{about}] Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        self.logger.info(msg)

    def log_test(self, fold, acc, precision, recall, f1, auc, ap, fpr, fnr, tpr, tnr):
        msg = (
            f"[test fold {fold}] "
            f"Acc:{acc:.4f} | Prec:{precision:.4f} | Rec:{recall:.4f} | "
            f"F1:{f1:.4f} | AUC:{auc:.4f} | AP:{ap:.4f} | "
            f"FPR:{fpr:.4f} | FNR:{fnr:.4f} | TPR:{tpr:.4f} | TNR:{tnr:.4f}"
        )
        self.logger.info(msg)

    def log_summary(self, mean_acc, std_acc):
        self.logger.info(f"[summary] Mean Acc:{mean_acc:.4f} | Std Acc:{std_acc:.4f}")