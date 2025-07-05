def compute_local_prototypes(model, dataloader, device='cuda'):
    model.eval()
    prototypes = {}
    counts = {}
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.feature_extractor(images)
            for i, label in enumerate(labels):
                label = label.item()
                feat = features[i].cpu().numpy()
                if label not in prototypes:
                    prototypes[label] = np.zeros_like(feat)
                    counts[label] = 0
                prototypes[label] += feat
                counts[label] += 1
    for label in prototypes:
        prototypes[label] /= counts[label]
    return prototypes


def select_global_prototypes(client_prototypes_list, lambda_=0.7, delta=10, tau=0.1):
    """
    Multi-objective optimization to select global prototypes
    """
    all_labels = set()
    for p in client_prototypes_list:
        all_labels.update(p.keys())
    all_labels = sorted(list(all_labels))

    # Initialize global prototypes
    global_prototypes = {l: np.zeros_like(next(iter(client_prototypes_list[0].values())) for l in all_labels}

    # Aggregate with contrastive loss objective
    for label in all_labels:
        local_reps = [p[label] for p in client_prototypes_list if label in p]
        if len(local_reps) == 0:
            continue
        mean_rep = np.mean(local_reps, axis=0)
        similarities = [F.cosine_similarity(torch.tensor(mean_rep), torch.tensor(r)).item() for r in local_reps]
        weights = F.softmax(torch.tensor(similarities) * delta, dim=0).cpu().numpy()
        weighted_avg = np.average(local_reps, axis=0, weights=weights)
        global_prototypes[label] = weighted_avg

    return global_prototypes