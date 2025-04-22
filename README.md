# 🧠 Master's Thesis Summary  
**October 2023**  
**Title:** Automated Optimization of DNN Models for People Counting with Infrared Sensors Using NAS

---

## 🏆 Publication

**“HW-SW Optimization of DNNs for Privacy-preserving People Counting on Low-resolution Infrared Arrays”**  
Seyed Morteza Mollaei, Daniele Jahier Pagliari, Chen Xie, Matteo Risso, Francesco Daghero, Alessio Burrello  
📍 *Presented at* **Design, Automation and Test in Europe (DATE), November 2023**  
🏅 **Selected as Best Paper of the Year**  
📄 [arXiv:2402.01226 [cs.LG]](https://arxiv.org/abs/2402.01226)

This paper presents a novel HW-SW co-design methodology using Neural Architecture Search (NAS) to optimize deep neural networks for real-time, privacy-preserving people counting on ultra-low-resolution infrared sensors. The work significantly advances efficiency and deployment feasibility in low-power IoT environments.

---

## 📌 Introduction

Neural Architecture Search (NAS) is gaining attention for automatically designing accurate deep learning models. However, applying NAS to **non-standard tasks** such as **people counting using ultra-low-resolution infrared (IR) sensors** has not been widely explored.

In this work, we apply **Pruning In Time (PIT)** — a NAS tool — to optimize deep learning architectures on an IR sensor dataset. We demonstrate that these models not only **improve accuracy by up to 2.85%**, but also **reduce memory usage by up to 61.6%** at iso-accuracy, enabling real-time operation on low-power IoT nodes.

---

## 🛠️ Proposed Method

We benchmarked PIT-NAS on 8 state-of-the-art architectures, tailored for the **LINAIGE dataset** (IR frames of size 8x8). Several configurations were tested:

- **Single-frame CNN**  
  Lightweight but does not use temporal info → lower accuracy.
  
- **Multi-frame CNN**  
  Takes a window of frames as multi-channel input → better accuracy but higher computation.

- **Temporal Convolutional Network (TCN)**  
  Handles time-series via dilation. Good for long sequences, but suffers when pruned.

- **Majority-Voting CNN**  
  Combines predictions from W frames → low complexity with high performance.

- **SuperNet + PIT**  
  Two-stage NAS: coarse architecture design via SuperNet, followed by fine-grained optimization via PIT.

---

## 📈 Results

We visualize **Pareto-optimal models** by plotting **balanced accuracy vs number of parameters**.

- **Majority-Voting CNN**:  
  Achieved **iso-accuracy with 62% fewer parameters**.  
  Also improved accuracy by 3.65% with 14.4% smaller size than the original.

- **SuperNet-small vs SuperNet-medium**:  
  SuperNet-medium attains 10.1% less balanced accuracy than the seed but requiring 90.5% less parameters.

- Compared to traditional techniques:  
  **Accuracy ↑ by 2.85%**,  
  **Memory ↓ by up to 44%**,  
  Enabling **real-time, energy-efficient** deployment.

---
