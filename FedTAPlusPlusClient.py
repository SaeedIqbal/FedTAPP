from transformers import ViTModel


class FedTAPlusPlusClient(nn.Module):
    def __init__(self, vit_backbone='google/vit-tiny-patch16-224', num_classes=100):
        super().__init__()
        self.feature_extractor = ViTModel.from_pretrained(vvit_backbone)
        self.tail_anchor = AdaptiveTailAnchor(feature_dim=192, num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x).last_hidden_state[:, 0, :]  # CLS token
        logits = self.tail_anchor(features)
        return logits

    def extract_features(self, x):
        return self.feature_extractor(x).last_hidden_state[:, 0, :]


# Main FL Loop
def fedta_plusplus_train(num_rounds=10, num_tasks=5, batch_size=32):
    datasets = ['CIFAR-100', 'ImageNet-R', 'NIH-ChestX-ray14', 'ODIR-100K']

    for round in range(num_rounds):
        print(f"\n--- Round {round + 1} ---")
        client_loaders = get_dataset_loaders(datasets[round % len(datasets)],
                                             root='/home/phd/datasets/',
                                             batch_size=batch_size,
                                             num_clients=5)
        client_models = []
        client_prototypes = []

        # Local Training
        for client_loader in client_loaders:
            client_model = FedTAPlusPlusClient().to('cuda')
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=3e-4)
            for epoch in range(2):  # 2 epochs per round
                for x, y in client_loader:
                    x, y = x.to('cuda'), y.to('cuda')
                    logits = client_model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            client_models.append(client_model)
            client_prototypes.append(compute_local_prototypes(client_model, client_loader))

        # Server Update
        global_prototypes = select_global_prototypes(client_prototypes)
        global_model = trajectory_aware_aggregation(global_model, client_models)

    print("Training Complete.")