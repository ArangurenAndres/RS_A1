import torch
import torch.nn as nn
import json

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[32, 16, 8]):
        super(NCF, self).__init__()
        print("Initializing NCF model with the following parameters:")
        print(f"num_users: {num_users}")
        print(f"num_items: {num_items}")
        print(f"embedding_dim: {embedding_dim}")
        print(f"mlp_layers: {mlp_layers}")

        # Define the two sets of embeddings
        # Embedding layers for GMF branch
        self.gmf_user_embed = nn.Embedding(num_users, embedding_dim) # user embeddings for GMF
        self.gmf_item_embed = nn.Embedding(num_items, embedding_dim) # item embeddings for GMF
        # Embedding layers for MLP branch
        self.mlp_user_embed = nn.Embedding(num_users, embedding_dim) # user embeddings for MLP
        self.mlp_item_embed = nn.Embedding(num_items, embedding_dim) # item embeddings for MLP

        mlp_input_size = embedding_dim * 2 # concatenate user and item
         #Builds the MLP branch of the Neural Collaborative Filtering model.
        mlp_modules = [] # List to hold the MLP layers
        for layer_size in mlp_layers:
            
            mlp_modules.append(nn.Linear(mlp_input_size, layer_size)) # Layer from inpout size to hidden_size
            mlp_modules.append(nn.ReLU()) # After the Fully connected layer apply a ReLU activation function
            mlp_input_size = layer_size
        self.mlp_layers = nn.Sequential(*mlp_modules) # Wrap all the layers into a single sequential module
        # Calculate the dimension for the final output layer 
        fusion_input_dim = embedding_dim + mlp_layers[-1] # We can do: embedding dimension + last dimension of last MLP layer
        self.output_layer = nn.Linear(fusion_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        gmf_user = self.gmf_user_embed(user_indices)
        gmf_item = self.gmf_item_embed(item_indices)
        gmf_output = gmf_user * gmf_item

        mlp_user = self.mlp_user_embed(user_indices)
        mlp_item = self.mlp_item_embed(item_indices)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        fusion = torch.cat([gmf_output, mlp_output], dim=-1) # Fusion layer to combine GMF and MLP outputs
        logits = self.output_layer(fusion) # Final prediction is a single output passed through sigmoid
        output = self.sigmoid(logits) 
        return output.squeeze()


if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_layers = config['num_layers']
    embedding_dim = config['embedding_dim']

    model = NCF(num_users=1000, num_items=1700,embedding_dim=embedding_dim,num_layers=num_layers)
    print(model)
