"""
Small utility functions for monitoring training performance
"""

import torch


class AverageMeter:
    """
    Keeps track of the average value of a metric (e.g., loss or accuracy)
    across multiple batches without storing every individual value in a
    list and computing the mean at the end
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        # value: metric value for the current batch (e.g., average batch loss)
        # n: number of samples in the batch (used for weighted averaging)
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


@torch.no_grad()
def accuracy(outputs, labels):
    """
    Computes the classification accuracy for a batch

    outputs: Tensor of shape (batch_size, num_classes) containing the
             unnormalized class scores (logits) predicted by the model
    labels:  Tensor of shape (batch_size,) containing the ground-truth
             class for each image

    argmax(dim=1) selects, for each image in the batch, the index of the
    class with the highest score, which corresponds to the model's prediction
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return 100.0 * correct / labels.size(0)