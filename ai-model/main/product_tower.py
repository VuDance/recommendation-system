import torch

class ProductTower(torch.nn.Module):
    def __init__(self, brand_vocab_size, text_dim=384, hidden_dim=128, output_dim=64):
        super(ProductTower, self).__init__()
        self.brand_embedding = torch.nn.Embedding(brand_vocab_size, 16)
        self.text_encoder = torch.nn.Linear(text_dim, 64)
        self.price_encoder = torch.nn.Linear(1, 16)
        self.combined_encoder = torch.nn.Sequential(
            torch.nn.Linear(64 + 16 + 16, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.LayerNorm(output_dim)
        )

    def forward(self, text_features, brand_ids, prices):
        text_encoded    = self.text_encoder(text_features)
        brand_embedded  = self.brand_embedding(brand_ids)
        price_encoded   = self.price_encoder(prices.unsqueeze(-1))
        combined        = torch.cat([text_encoded, brand_embedded, price_encoded], dim=-1)
        output          = self.combined_encoder(combined)
        return torch.nn.functional.normalize(output, p=2, dim=-1)