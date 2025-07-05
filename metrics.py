import numpy as np
import torch
from sklearn.metrics import accuracy_score


def compute_online_forgetting_rate(acc_history):
    """
    Online Forgetting Rate (OFR): Average degradation over time.
    acc_history: list of accuracy values per task
    """
    ofr = 0.0
    n_tasks = len(acc_history)
    for t in range(n_tasks):
        best_acc = max(acc_history[:t+1])
        curr_acc = acc_history[t]
        ofr += (best_acc - curr_acc)
    return ofr / n_tasks


def compute_drift_robustness_score(task_accuracies):
    """
    Drift Robustness Score (DRS): Stability across task transitions.
    task_accuracies: list of accuracies per task
    """
    drs = 0.0
    for i in range(1, len(task_accuracies)):
        delta = abs(task_accuracies[i] - task_accuracies[i-1]) / task_accuracies[i-1]
        drs += (1 - delta)
    return drs / (len(task_accuracies) - 1)


def compute_ood_detection_accuracy(logits, labels, threshold=0.45):
    """
    Out-of-Distribution Detection Accuracy (ODA).
    """
    probs = torch.softmax(logits, dim=1)
    max_probs = probs.max(dim=1).values
    ood_mask = (max_probs < threshold).cpu().numpy()
    true_ood = (labels == -1).cpu().numpy()  # Assume -1 is OOD label
    tp = np.sum(np.logical_and(ood_mask, true_ood))
    tn = np.sum(np.logical_and(~ood_mask, ~true_ood))
    total = len(labels)
    return (tp + tn) / total


def compute_temporal_consistency_loss(global_prototypes, local_prototypes):
    """
    Temporal Consistency Loss (TCL): Measures prototype shift over time.
    """
    losses = []
    for cls in global_prototypes:
        if cls in local_prototypes:
            loss = torch.norm(global_prototypes[cls] - local_prototypes[cls], p=2)
            losses.append(loss.item())
    return np.mean(losses) if losses else float('inf')