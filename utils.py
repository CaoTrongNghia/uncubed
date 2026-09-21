import torch
import ase.io

def prepare_molecule(file_path, device="cpu"):
    atoms = ase.io.read(file_path)
    
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    batch = torch.zeros(len(z), dtype=torch.long, device=device)

    return z, pos, batch
