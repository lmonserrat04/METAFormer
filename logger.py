import logging
import os

# Setup logging
log_path = "training_log.txt"
logging.basicConfig(
    filename=log_path,
    filemode='w',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

def log_losses(stage, epoch, train_loss, val_loss):
    msg = f"[{stage}] Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
    
    
    
    logging.info(msg)
    print(msg)