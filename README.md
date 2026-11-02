# uncubed

**uncubed** is a high-precision deep learning surrogate engine designed to accelerate quantum chemistry workflows. Powered by an Equivariant Transformer neural network, **uncubed** circumvents the steep $O(N^3)$ computational bottleneck of classical Density Functional Theory (DFT).

It serves as a rapid virtual screening tool to evaluate large molecular libraries, enabling researchers to filter candidate molecules for target energy profiles in seconds before executing full DFT simulations.

*Note: uncubed is designed as a fast predictive filter to complement, rather than completely replace, first-principles DFT calculations.*

### Why choose uncubed?

* **Sub-Cubic Efficiency:** Bypasses $O(N^3)$ scaling limits to evaluate molecular energy landscapes and atomic forces in seconds.
* **Sub-Chemical Precision:** Achieves an MAE of 0.01 across potential energy states and atomic forces—surpassing standard chemical accuracy thresholds (~1 kcal/mol or ~0.043 eV) by 4x.
* **Open-Source & Modular:** Built on a PyTorch architecture designed for integration into computational chemistry and drug discovery pipelines.

## Quickstart

### 1. Clone the repository
```bash
git clone [https://github.com/CaoTrongNghia/uncubed.git](https://github.com/CaoTrongNghia/uncubed.git)
cd uncubed
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run energy & force prediction on a sample structure
Run inference on the included `ethanol.xyz` test structure:

```bash
python predict.py --input sample_data/ethanol.xyz
```

#### Expected Output
```text
Loading uncubed model weights from: weights/best_model.pt
Processing molecule: sample_data/ethanol.xyz

=================== UNCUBED INFERENCE ===================
 Predicted Potential Energy : -154.284019 eV
 Max Atomic Force Component : 0.041203 eV/Å
=========================================================
```

---

## Model & Training Highlights

* **Architecture:** Equivariant Transformer (TorchMD-Net)
* **Dataset:** SPICE 1.1.4 dataset (Sharded data loader pipeline)
* **Target Quantities:** Total Potential Energy ($E$) and Atomic Force Vectors ($F = -\nabla E$) via autograd
* **Hyperparameters:** 6 layers, 128 embedding dimension, 8 attention heads, 32 Gaussian RBFs, 5.0 Å cutoff radius

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
