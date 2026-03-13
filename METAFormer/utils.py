
import numpy as np
import torch


from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
from logger import Logger



class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
        from https://github.com/gzerveas/mvts_transformer
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.mse_loss(masked_pred, masked_true)









def test(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            aal, cc200, do160 = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device)
            targets = targets.to(device)
            outputs = model(aal, cc200, do160)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            y_true.extend(np.argmax(targets.detach().cpu().numpy(), axis=1))
            y_pred.extend(np.argmax(np.where(probs > 0.5, 1, 0), axis=1))
            y_prob.extend(probs[:, 1])
    return y_true, y_pred, y_prob