import torch
from torchmdnet.models.model import create_model

# Custom Dictionary that absorbs KeyErrors during TorchMD-Net initialization
class Config(dict):
    def __getitem__(self, key):
        if key not in self:
            if key == 'atom_filter': return -1
            if key == 'layernorm_on_vec': return 'whitened'
            if key == 'aggr': return 'add'
            return None
        return super().__getitem__(key)

# Exact architecture parameters from your training notebook
UNCUBED_CONFIG = Config({
    'model': 'equivariant-transformer',
    'embedding_dimension': 128,
    'num_layers': 6,
    'num_rbf': 32,
    'rbf_type': 'gauss',
    'trainable_rbf': False,
    'activation': 'silu',
    'attn_activation': 'silu',
    'num_heads': 8,
    'distance_influence': 'both',
    'cutoff_lower': 0.0,
    'cutoff_upper': 5.0,
    'max_num_neighbors': 32,
    'neighbor_embedding': True,
    'derivative': True,          # Calculates forces via autograd (-dE/dx)
    'max_z': 100,
    'output_model': 'Scalar',
    'precision': 32,
    'reduce_op': 'add',
    'prior_model': None,
    'atom_filter': -1,
    'layernorm_on_vec': 'whitened',
    'aggr': 'add'
})

def load_uncubed_model(weights_path, device="cpu"):
    """
    Instantiates the Equivariant Transformer architecture and loads trained weights.
    """
    # 1. Rebuild base architecture
    model = create_model(UNCUBED_CONFIG).to(device)

    # 2. Load checkpoint dictionary
    checkpoint = torch.load(weights_path, map_location=device)

    # 3. Extract model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model
