import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductTower(nn.Module):
    def __init__(self, 
                 product_id,
                 product_title,
                 product_description,
                 product_brand,
                 product_id_dim = 64,
                 product_title_dim = 64,
                 product_description_dim = 64,
                 product_brand_dim = 16,
                 hidden_dim = 128,
                 output_dim = 64):
        super(ProductTower, self).__init__()
        self.product_id_embedding = nn.Embedding(product_id, product_id_dim)
        self.product_title_embedding = nn.Embedding(product_title, product_title_dim)
        self.product_description_embedding = nn.Embedding(product_description, product_description_dim)
        self.product_brand_embedding = nn.Embedding(product_brand, product_brand_dim)
        