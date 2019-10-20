"""
Use a composite loss of weighted-cross entropy and dice loss proposed in https://arxiv.org/pdf/1801.04161.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os.path
from util import estimate_weights, make_one_hot

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


def dice_loss(pred, encoded_target):
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
    
    loss_per_channel = 1 - (numerator / denominator)
    
    loss = loss_per_channel.sum() / output.size(1)
    del output, encoded_target
    
    return loss.mean()


def cross_entropy_loss(pred, target, weight):
    """
    :param pred : N x C x H x W
    :param target : N x H x W
    :param: weight : N x H x W
    
    """
    
    loss_func = nn.CrossEntropyLoss()
    
    loss = loss_func(pred, target)
        
    return torch.mean(torch.mul(loss, weight))

def combined_loss(pred, target, device, n_classes):
    """
    :param pred: N x C x H x W
    :param target: N x H x W
    """
    
    weights = estimate_weights(target.float())
    weights = weights.to(device)
       
    cross = cross_entropy_loss(pred, target, weights)
    
    target_oh = make_one_hot(target.long(), n_classes, device)
    
    dice = dice_loss(pred, target_oh)
    
    loss = cross + dice
    
    del weights
    
    return loss, cross, dice