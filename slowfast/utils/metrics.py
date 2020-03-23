#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""
import numpy as np
import torch
import sys
import os
from sklearn.utils.extmath import softmax
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
def softmax22(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    #preds_numpy = preds.clone()
    #propability = np.transpose(np.array(softmax(preds_numpy.cpu().numpy())))
    #print(propability.shape)
    #print(propability)
    #print(propability[21])
    #cwd = os.getcwd()
    #tmp_dir = os.path.join(cwd, "tmp/probability.npy")
    #jogging_label = 21
    #np.save(tmp_dir,propability[jogging_label])     
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    
    #print(_top_max_k_vals[:,168])
    #print(top_max_k_inds)
    #print(rep_max_k_labels)
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    #clone_pred = top_max_k_inds.clone()
    #clone_label = rep_max_k_labels.clone()
    #clone_pred[top_max_k_inds!=168]=0
    #clone_label[rep_max_k_labels!=168]=0
    #print("#####")
    #print(clone_pred[0])
    #print(clone_label[0])
    #top_binary_correct = clone_pred.eq(clone_label)
    
    #print(top_binary_correct[0])
    #binary_accuracy = (torch.sum((top_binary_correct[0] == True).float())/(list(top_binary_correct[0].size())[0])).data.cpu().numpy()
    #print("Binary accuracy: ",binary_accuracy)
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]
