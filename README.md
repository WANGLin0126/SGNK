


# Simplifying Graph Neural Kernels:  from Stacking Layers to Collapsed Structure

[🔗 Project Page (Anonymous Open Access)](https://anonymous.4open.science/r/SGNK-1CE4/)

## 🧠 Introduction

This repository provides the official implementation of:

- **SGTK (Simplified Graph Neural Tangent Kernel)**
- **SGNK (Simplified Graph Neural Kernel)**

These methods are designed to enhance the efficiency and scalability of graph learning while preserving the expressive power of Graph Neural Networks (GNNs).

While traditional **Graph Neural Tangent Kernel (GNTK)** bridges kernel methods and GNNs, it suffers from high computational complexity due to redundant layer-stacking operations. **SGTK** and **SGNK** address these limitations by introducing more efficient kernel computation strategies.

---

## 🚀 Key Contributions

### 🔷 SGTK: Simplified Graph Neural Tangent Kernel

- Replaces deep layer stacking with a **continuous $K$-step aggregation** process.
- Eliminates redundant computations in the original GNTK.
- Achieves strong expressive power with significantly improved computational efficiency.

### 🔷 SGNK: Simplified Graph Neural Kernel

- Models **infinitely wide GNNs as Gaussian Processes**.
- Computes kernel values directly from the expected outputs of activation functions.
- Avoids explicit layer-by-layer computation entirely.
- Further reduces computational overhead while capturing complex graph structures.

---

## 🧪 Experimental Results

We evaluate SGTK and SGNK on standard **node classification** and **graph classification** benchmarks. Results show that:

- Both methods achieve **comparable accuracy** to state-of-the-art GNN and kernel methods.
- They offer **significant improvements in computational efficiency**, making them suitable for large-scale graph data.

---

## 📁 Repository Structure

```bash
Graphs_classification/
├── dataset/              # graph datasets
├── outputs/              # Output files for experiments 
├── gram.py               # kernel matrix computation and saving
├── run_gram.sh           # graph kernel implementation
├── run_search.sh         # svm searching implementation
├── search.py             # svm searching implementation
├── sgnk.py               # graph SGNK implementation
├── sgtk.py               # graph SGTK implementation
└── util.py               # Utility functions
Nodes_classification/
├── datasets/             # node datasets
├── outputs/              # Output files for experiments 
├── LoadData.py           # Data loading and preprocessing
├── run_search_krr.sh     # kernel ridge regression implementation
├── run_search_svm.sh     # svm implementation
├── search_krr.py         # kernel ridge regression implementation
├── search_svm.py         # svm implementation
├── sgnk.py               # SGNK implementation
├── sgtk.py               # SGTK implementation
└── utils/                # Utility functions
requirements.txt      # Python dependencies
README.md             # Project documentation

```

---

## ⚙️ Installation & Usage

### 📦 Install Dependencies

```bash
git clone https://anonymous.4open.science/r/SGNK-1CE4/
cd SGNK
pip install -r requirements.txt
```

### 🚀 Run Examples

```bash
# Run SGTK on the Cora datasets
cd Nodes_classification
python run_search_svm.py --dataset Cora --kernel SGTK --K 3

# Run SGTK on the MUTAG dataset
cd Graphs_classification
python gram.py --dataset MUTAG --kernel SGTK --K 3
python search.py --dataset MUTAG --kernel SGTK --K 3
```

---

## 📊 Supported Datasets

- **Node classification**: `Cora`, `Citeseer`, `Pubmed`,`Photo`, `Computers`.
- **Graph classification**: `IMDBBINARY`, `IMDBMULTI`, `MUTAG`,`PTC`.

<!-- --- -->

<!-- ## 📜 Citation

If you find this work helpful, please cite:

```bibtex
@article{sgnk2024,
  title={Simplified Graph Neural Kernel: Efficient and Expressive Graph Learning},
  author={Anonymous},
  journal={Preprint},
  year={2024},
  url={https://anonymous.4open.science/r/SGNK-1CE4/}
}
``` -->

---

## 📬 Contact

For questions or collaborations, please visit the [project page](https://anonymous.4open.science/r/SGNK-1CE4/).

---

## 📝 License

This project is released under an open-source license. See the `LICENSE` file for details.

License: [MIT](./LICENSE) © 2025 WANG Lin