def detect_ood_samples(embedding, global_prototypes, threshold=0.45, tau=0.1, K=5):
    embedding = F.normalize(embedding, p=2, dim=-1)
    scores = []
    prototype_embeddings = torch.tensor(np.stack(list(global_prototypes.values())))
    prototype_embeddings = F.normalize(prototype_embeddings, p=2, dim=-1)

    for emb in embedding:
        sim = F.cosine_similarity(emb.unsqueeze(0), prototype_embeddings)
        max_sim = sim.max().item()
        if max_sim < threshold:
            # Use contrastive outlier detection
            neg_samples = torch.rand(K, prototype_embeddings.shape[1]).to(embedding.device)
            logits = torch.cat([sim.unsqueeze(0), F.cosine_similarity(emb.unsqueeze(0), neg_samples)])
            probs = F.log_softmax(logits / tau, dim=0)
            ood_score = probs[0].item()
            if ood_score < threshold:
                yield True
        yield False