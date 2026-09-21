import argparse
import torch
from uncubed.model import load_uncubed_model
from uncubed.utils import prepare_molecule

def main():
    parser = argparse.ArgumentParser(description="uncubed: Neural Surrogate Engine for Quantum DFT Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to input .xyz file")
    parser.add_argument("--weights", type=str, default="weights/best_model.pt", help="Path to weights file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading weights from: {args.weights}")
    model = load_uncubed_model(args.weights, device=device)

    print(f"Processing molecule: {args.input}")
    z, pos, batch = prepare_molecule(args.input, device=device)

    # Autograd required for force vectors (-dE/dPos)
    pos.requires_grad_(True)
    pred_energy, pred_forces = model(z, pos, batch=batch)

    print("\n=================== UNCUBED INFERENCE ===================")
    print(f" Predicted Potential Energy : {pred_energy.item():.6f} eV")
    print(f" Max Atomic Force Component : {pred_forces.abs().max().item():.6f} eV/Å")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
