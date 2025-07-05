# aggregation.py
import torch
import numpy as np


def procrustes_alignment(A, B):
    """
    Soft Procrustes analysis for rotation matrix alignment.
    """
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(B_np.T @ A_np)
    R = U @ Vt
    return torch.tensor(R, device=A.device, dtype=A.dtype)


def trajectory_aware_aggregation(global_model, client_models, weights=None):
    """
    Aligns client updates before federated averaging.
    """
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)

    aligned_states = []
    global_state = {k: v.clone() for k, v in global_model.named_parameters()}

    for client in client_models:
        state = client.state_dict()
        aligned_state = {}

        for name, param in state.items():
            g_param = global_state[name]
            if param.shape != g_param.shape:
                aligned_state[name] = param
                continue
            R = procrustes_alignment(param, g_param)
            aligned_state[name] = torch.tensor(R @ param.cpu().numpy(), device=param.device)

        aligned_states.append(aligned_state)

    averaged_state = {}
    for key in global_state:
        avg_tensor = torch.mean(
            torch.stack([state[key] * w for state, w in zip(aligned_states, weights)], dim=0),
            dim=0
        )
        averaged_state[key] = avg_tensor

    global_model.load_state_dict(averaged_state)
    return global_model