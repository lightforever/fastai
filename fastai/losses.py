from .imports import *
from .torch_imports import *

def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9):
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)
    precision = tp / (y_pred.sum(dim=1)+eps)
    recall = tp / (y_true.sum(dim=1)+eps)
    return torch.mean(
        precision*recall / (precision*(beta**2)+recall+eps) * (1+beta**2))

class Bce_Dice(nn.Module):
    def __init__(self, dice_weight=1):
        super(Bce_Dice, self).__init__()
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-15
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= torch.log(2 * intersection / union)

        return loss
