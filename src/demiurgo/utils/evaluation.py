"""
Complete evaluation of a trained model on a dataset

Computes overall accuracy, per-class metrics (precision, recall, f1-score),
and the confusion matrix
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate_model(model, loader, device, class_names):
    """
    Runs the entire data loader through the model (without computing gradients,
    in evaluation mode) and computes all metrics in a single pass

    Returns a dictionary containing: accuracy, confusion_matrix,
    classification_report (dict with precision/recall/f1/support per class)
    and classification_report_text (formatted string for printing or saving to a .txt file)
    """
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = 100.0 * (all_preds == all_labels).mean()

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(class_names))),
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }