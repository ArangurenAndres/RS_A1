import torch
import torch.nn as nn
import json

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[32, 16, 8], dropout_rate=0.2):
        super(NCF, self).__init__()
        print("Initializing NCF model with the following parameters:")
        print(f"num_users: {num_users}")
        print(f"num_items: {num_items}")
        print(f"embedding_dim: {embedding_dim}")
        print(f"mlp_layers: {mlp_layers}")
        print(f"dropout_rate: {dropout_rate}")

        # GMF Embedding Layers
        self.gmf_user_embed = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embed = nn.Embedding(num_items, embedding_dim)

        # MLP Embedding Layers
        self.mlp_user_embed = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embed = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.gmf_user_embed.weight)
        nn.init.xavier_uniform_(self.gmf_item_embed.weight)
        nn.init.xavier_uniform_(self.mlp_user_embed.weight)
        nn.init.xavier_uniform_(self.mlp_item_embed.weight)

        # MLP Layers
        mlp_input_size = embedding_dim * 2
        mlp_modules = []
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input_size, layer_size))
            mlp_modules.append(nn.BatchNorm1d(layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout_rate))
            mlp_input_size = layer_size
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Fusion layer (GMF + MLP output)
        fusion_input_dim = embedding_dim + mlp_layers[-1]
        self.output_layer = nn.Linear(fusion_input_dim, 1)

    def forward(self, user_indices, item_indices):
        # GMF part
        gmf_user = self.gmf_user_embed(user_indices)
        gmf_item = self.gmf_item_embed(item_indices)
        gmf_output = gmf_user * gmf_item  # element-wise product

        # MLP part
        mlp_user = self.mlp_user_embed(user_indices)
        mlp_item = self.mlp_item_embed(item_indices)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # Merge GMF + MLP
        fusion = torch.cat([gmf_output, mlp_output], dim=-1)
        logits = self.output_layer(fusion)
        return logits.squeeze()  # Used with BCEWithLogitsLoss

# Optional: run test
if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = NCF(
        num_users=1000,
        num_items=1700,
        embedding_dim=config['embedding_dim'],
        mlp_layers=config['layers'],
        dropout_rate=config['dropout_rate']
    )

    print(model)
