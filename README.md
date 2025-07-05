# 📦 FedTA++: A Temporal-Aware Framework for Federated Continual Learning

This repository contains the official implementation of **FedTA++**, a novel framework for **Federated Continual Learning (FCL)** that effectively addresses **spatial-temporal catastrophic forgetting** under extreme data heterogeneity. This work extends the original **FedTA** method by introducing **Temporal-Aware Feature Stabilization**, **Dynamic Global Prototype Selection**, **Contrastive Outlier Detection for OOD Tasks**, and **Trajectory-Aware Aggregation**, enabling robust knowledge retention and adaptation in real-world FCL scenarios.

## 🔍 Overview

**FedTA++** is designed to handle both **spatial heterogeneity** (non-IID data across clients) and **temporal heterogeneity** (concept drift and unpredictable task evolution), which are major challenges in real-world deployment of federated learning systems.

The framework combines four key components:

1. **Temporal-Aware Feature Stabilization via Adaptive Tail Anchors**: dynamically adjusts feature embeddings based on concept drift estimation.
2. **Dynamic Global Prototype Selection with Temporal Consistency**: selects prototypes that evolve smoothly over time, ensuring global model consistency.
3. **Contrastive Outlier Detection for OOD Task Handling**: identifies and incorporates novel semantic concepts without memory replay or pseudo-data generation.
4. **Trajectory-Aware Aggregation with Memory of Client Trajectories**: aligns client updates using historical learning paths before aggregation.

These innovations allow **FedTA++** to outperform existing state-of-the-art methods such as **FedAvg**, **FedLwF**, **GLFC**, and **FedMGP**, especially under dynamic, open-ended continual learning settings.

---

## 🧠 Based Paper

- **Title:** *Handling Spatial-Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor*
- **Authors:** Yu, H., Yang, X., Zhang, L., Gu, H., Li, T., Fan, L., & Yang, Q.
- **Conference:** Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)
- **Year:** 2025
- **Pages:** 4874–4883

### Paper Abstract Summary

Federated continual learning (FCL) enables clients to learn from sequential tasks while preserving data privacy. However, spatial-temporal data heterogeneity causes severe parameter-forgetting and output-forgetting. To address this, the paper proposes **FedTA**, which mixes trainable tail anchors with frozen features to maintain class-wise stability. The original FedTA introduces input enhancement, selective knowledge fusion, and best prototype selection.

Our extension, **FedTA++**, builds upon FedTA by introducing temporal modeling, contrastive outlier detection, and trajectory-aware aggregation to enhance adaptability under extreme temporal shifts and OOD tasks.

> **Citation**
```bibtex
@inproceedings{yu2025fedta,
  title={Handling Spatial-Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor},
  author={Yu, Hao and Yang, Xin and Zhang, Le and Gu, Hanlin and Li, Tianrui and Fan, Lixin and Yang, Qiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4874--4883},
  year={2025}
}
```

---

## 📁 Datasets

We evaluate FedTA++ on four diverse datasets representing various domains and levels of heterogeneity:

### 1. **CIFAR-100**
- **Description:** 60,000 32×32 RGB images across 100 classes.
- **Usage:** Simulates continual learning with local class exclusivity.
- **Link:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

### 2. **ImageNet-R**
- **Description:** 45,000 stylized renditions of 200 object classes from ImageNet.
- **Usage:** Evaluates visual diversity and domain shift resilience.
- **Link:** [https://huggingface.co/datasets/imagenet-r](https://huggingface.co/datasets/imagenet-r)

### 3. **NIH ChestX-ray14**
- **Description:** Over 112,000 frontal-view chest X-ray images labeled with up to 14 thoracic disease categories.
- **Usage:** Medical imaging benchmark with long-tailed distributions and label skew.
- **Link:** [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### 4. **ODIR-100K**
- **Description:** More than 100,000 retinal fundus photographs covering over 70 distinct eye disease labels.
- **Usage:** Real-world medical dataset with high inter-class variability.
- **Link:** [https://figshare.com/articles/dataset/ODIR-100K_dataset_of_eye_diseases_images_and_multi-label_annotations/19396253](https://figshare.com/articles/dataset/ODIR-100K_dataset_of_eye_diseases_images_and_multi-label_annotations/19396253)

All datasets are assumed to be located at `/home/phd/datasets/`. You can modify this path in `data_loader.py`.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/fedta-plusplus.git
cd fedta-plusplus
pip install -r requirements.txt
```

### Requirements:
- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- transformers
- pandas, numpy, scikit-learn
- torch>=2.0
- torchvision>=0.15
- transformers>=4.30
- scikit-learn
- numpy
- matplotlib

---

## 🚀 Usage

### 1. Dataset Setup
Place all datasets under the following directory structure:

```
/home/phd/datasets/
├── cifar-100/
├── imagenet-r/
├── nih_chestxray14/
└── odir_100k/
```

You may need to preprocess NIH ChestX-ray14 and ODIR-100K with custom label files or use provided CSVs.

### 2. Train the Model

```bash
python train.py --dataset CIFAR-100 --num_clients 5 --num_rounds 20 --batch_size 32
```

### 3. Evaluate Performance

```bash
python eval.py --model_path ./saved_models/fedta_plusplus.pth
```

---

## 📊 Evaluation Metrics

The code supports evaluation using the following **advanced FCL metrics**:

| Metric | Description |
|--------|-------------|
| **OFR (Online Forgetting Rate)** ↓ | Measures degradation of performance on previous tasks |
| **DRS (Drift Robustness Score)** ↑ | Evaluates stability across concept drift points |
| **ODA (OOD Detection Accuracy)** ↑ | Measures how well the system detects novel semantic concepts |
| **TCL (Temporal Consistency Loss)** ↓ | Tracks instability in global prototypes over time |

---

## 📌 Methodology Highlights

### 1. **Adaptive Tail Anchors**
Tail anchors dynamically adjust feature embeddings based on concept drift, reducing both parameter-forgetting and output-forgetting.

### 2. **Dynamic Global Prototype Selection**
Selects stable and discriminative global prototypes using a contrastive loss-based multi-objective optimization.

### 3. **Contrastive Outlier Detection**
Detects and integrates novel OOD concepts without relying on memory replay.

### 4. **Trajectory-Aware Aggregation**
Aligns client updates using Procrustes-style rotation matrices to reduce misalignment during fusion.

---

## 📈 Results

Extensive experiments show that **FedTA++ consistently surpasses all current SOTA methods** in accuracy, drift robustness, and OOD detection, while maintaining low communication cost and strong privacy guarantees.

Key results include:
- **95% accuracy on CIFAR-100** with only **20 MB/round** communication and **ε = 0.05**.
- Strong generalization on **medical imaging datasets**, including **NIH ChestX-ray14** and **ODIR-100K**, where label scarcity and concept drift are common.

---

## 📌 License

MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For any questions, issues, or suggestions, please open an issue or contact us:

- **Lead Author:** saeed.iqbal@szu.edu.cn
- **Co-Author:** driachoudhry@gmail.com

---

## 📚 References

1. Yu, H., Yang, X., Zhang, L., Gu, H., Li, T., Fan, L., & Yang, Q. (2025). *Handling Spatial-Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor*. In CVPR (pp. 4874–4883).
2. Yang et al. (2024). *Federated Continual Learning via Knowledge Fusion: A Survey*. IEEE TKDE.
3. Dong et al. (2022). *Federated Class-Incremental Learning*. CVPR.
4. Zhang et al. (2023). *Target: Federated Class-Continual Learning via Exemplar-Free Distillation*. CVPR.
5. Sharshar et al. (2024). *Personalized Federated Continual Learning via Multi-Granularity Prompt*. KDD.

---

Thank you for your interest in **FedTA++**! We hope this repository serves as a valuable resource for researchers and practitioners working in **federated learning**, **continual learning**, and **privacy-preserving AI**.

Please star ⭐ the repo if you find it useful!
