
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_score(pred, encoded_target):
    """
    :param pred : N x C x H x W logits
    :param encoded_target : N x C x H x W LongTensor
    """
    
    output = F.softmax(pred, dim = 1)
    
    eps = 1
 
    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    
    loss_per_channel = numerator / denominator
    
    score = loss_per_channel.sum() / output.size(1)

    del output, encoded_target

    return score.mean()