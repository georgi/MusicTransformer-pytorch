import torch
import torch.nn.functional as F

def smooth_cross_entropy(prediction, target, eps=0.1, ignore_index=0):
    mask = (target == ignore_index).unsqueeze(-1)
    n_classes = prediction.shape[-1]
    p = F.one_hot(target, n_classes)
    u = 1.0 / n_classes
    p_prime = (1.0 - eps) * p + eps * u
    p_prime = p_prime.masked_fill(mask, 0)
    h = -torch.sum(p_prime * F.log_softmax(prediction, -1))
    n_items = torch.sum(target != ignore_index)
    return h / n_items


