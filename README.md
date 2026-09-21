# uncubed

**uncubed** is a high-precision deep learning surrogate engine designed to accelerate quantum chemistry workflows. Powered by an Equivariant Transformer neural network, **uncubed** circumvents the steep $O(N^3)$ computational bottleneck of classical Density Functional Theory (DFT).

It serves as a rapid virtual screening tool to evaluate large molecular libraries, enabling researchers to filter candidate molecules for target energy profiles in seconds before executing full DFT simulations.

*Note: uncubed is designed as an ultra-fast predictive filter to complement, rather than completely replace, first-principles DFT calculations.*

### Why choose uncubed?

* **Sub-Cubic Efficiency:** Bypasses $O(N^3)$ scaling limits to evaluate molecular energy landscapes and atomic forces in seconds.
* **Sub-Chemical Precision:** Achieves an MAE of 0.01 across potential energy states and atomic forces—surpassing standard chemical accuracy thresholds (~1 kcal/mol or ~0.043 eV) by 4x.
* **Open-Source & Modular:** Built on a PyTorch architecture designed for integration into computational chemistry and drug discovery pipelines.
