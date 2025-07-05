import torch
from torch.utils.data import DataLoader
from fedta import FedTAPlusPlusClient
from vit_backbone import ViTBackbone
from aggregation import trajectory_aware_aggregation
from metrics import compute_online_forgetting_rate, compute_drift_robustness_score, compute_ood_detection_accuracy, compute_temporal_consistency_loss


def get_dataloader(dataset_name):
    from data_loader import get_dataset_loaders
    return get_dataset_loaders(dataset_name, root='/home/phd/datasets/', batch_size=32)


def train_fedta_plusplus(num_rounds=10, num_clients=5, num_tasks=5):
    """
    Main federated continual learning loop for FedTA++
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = FedTAPlusPlusClient().to(device)
    client_loaders = [get_dataloader('CIFAR-100') for _ in range(num_clients)]

    client_models = [FedTAPlusPlusClient().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4) for model in client_models]

    acc_history = []

    for round in range(num_rounds):
        print(f"\n--- Federated Round {round + 1} ---")
        client_weights = []
        client_prototypes = []

        for i, loader in enumerate(client_loaders):
            model = client_models[i]
            optimizer = optimizers[i]
            model.train()

            prototypes = {}
            counts = {}

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()

                feats = model.extract_features(images)
                _, preds = torch.max(logits, 1)
                acc = accuracy_score(preds.cpu(), labels.cpu())
                acc_history.append(acc)

                # Update prototypes
                for feat, label in zip(feats, labels):
                    label = label.item()
                    if label not in prototypes:
                        prototypes[label] = torch.zeros_like(feat)
                        counts[label] = 0
                    prototypes[label] += feat
                    counts[label] += 1

            for label in prototypes:
                prototypes[label] /= counts[label]
            client_prototypes.append(prototypes)

        # Aggregate with trajectory-aware method
        global_prototypes = trajectory_aware_aggregation(global_model, client_models)
        print("Global Model Updated")

    print("Training Complete.")
    ofr = compute_online_forgetting_rate(acc_history)
    print(f"Online Forgetting Rate (OFR): {ofr:.4f}")