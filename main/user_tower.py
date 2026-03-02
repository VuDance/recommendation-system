import torch

class UserTower(torch.nn.Module):
    def __init__(self, product_vocab_size, hidden_dim=128, output_dim=64):
        super(UserTower, self).__init__()
        self.product_embedding = torch.nn.Embedding(product_vocab_size, 32)
        self.history_encoder = torch.nn.GRU(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.output_encoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim)
        )

    def forward(self, product_history):
        embedded = self.product_embedding(product_history)
        _, hidden = self.history_encoder(embedded)
        output = self.output_encoder(hidden[-1])
        return torch.nn.functional.normalize(output, p=2, dim=-1)