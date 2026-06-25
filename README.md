# HMQD-Net: Hierarchical Multimodal Representation and Query-Aware Distillation for UAV-Based Wheat Scab Detection

> Official repository for the paper **"Hierarchical multimodal representation and query-aware distillation for UAV-based wheat scab monitoring"**, submitted to *Engineering Applications of Artificial Intelligence (EAAI)*.

---

## 📌 Overview

**HMQD-Net** is a hierarchical multimodal representation and query-aware distillation network for **RGB-D (RGB + Depth)** wheat scab detection from unmanned aerial vehicle (UAV) imagery. Unmanned aerial vehicle remote sensing offers a feasible solution for large-scale wheat disease monitoring, yet existing UAV-based wheat scab detection methods still struggle with the small size of diseased spike targets, severe canopy occlusion, and the limited representational capacity of single-modal images.

To address these challenges, HMQD-Net is built on top of [**RT-DETR**](https://github.com/lyuwenyu/RT-DETR) and introduces a dual-branch, modality-specific design that explicitly handles the heterogeneity between appearance and geometric cues, performs selective cross-modal fusion, and refines the decoder queries at the decision level. The framework integrates four key components:

- **SSHR** — Sparse State-space Hierarchical Representation (RGB branch)
- **PDER** — Progressive Depth Expert Routing (depth branch)
- **DTF** — Dual-Stream Token-selective Fusion (cross-modal fusion)
- **QAKD** — Query-Aware Knowledge Distillation (decoder-level distillation)

HMQD-Net achieves **32.4% AP** and **68.7% AP50** on our self-constructed **WS-D** dataset, and generalizes well to the public **MSWDD** dataset (**76.8% AP**), while maintaining a favorable accuracy–efficiency trade-off and supporting real-time inference on resource-constrained edge devices.

---

## 🔔 Release Status

> **Important.** This repository is being prepared for public release. To comply with our institutional and journal policies during the peer-review process, **the full source code, configuration files, pretrained model weights, and the WS-D dataset will be made publicly available upon acceptance of the paper.**

We are committed to full reproducibility. This page already documents the complete method, training protocol, hyperparameters, dataset specification, and deployment procedure, so that the released code can be reproduced and verified without ambiguity. The repository structure, usage instructions, and result tables below reflect the final intended release.

| Resource | Status | Link |
| --- | --- | --- |
| Method description & documentation | ✅ Available (this page) | — |
| Source code (training / inference) | 🔒 Upon acceptance | — |
| Pretrained weights | 🔒 Upon acceptance | `https://XXX` *(to be updated)* |
| WS-D dataset (RGB-D) | 🔒 Upon acceptance | `https://XXX` *(to be updated)* |

> 📦 **Model weights and dataset download:** `https://XXX` *(placeholder — to be replaced)*

---

## ✨ Highlights

- **A unified RGB-D framework for UAV-based wheat scab detection.** Unlike methods relying on a shared backbone and simple channel-level fusion, HMQD-Net explicitly addresses modality heterogeneity between appearance and geometric cues.
- **Modality-specific hierarchical encoding.** SSHR strengthens long-range contextual modeling of disease appearance in the RGB branch via sparse region-level state-space scanning, while PDER adaptively captures disease-relevant geometric structures in the depth branch via a progressive mixture-of-experts routing mechanism.
- **Selective cross-modal fusion.** DTF performs multi-scale channel alignment and token-selective attention, suppressing redundant background interactions instead of uniform pixel-wise fusion.
- **Decision-level query distillation.** QAKD extends the improvement from the encoding stage to the decoder by selectively distilling high-confidence effective queries, improving small-target detection **without increasing inference cost**.
- **A new UAV-based RGB-D wheat scab dataset (WS-D).** 1,950 spatially aligned RGB-D image pairs with 6,799 annotated diseased-spike instances, collected across three wheat-growing regions in Anhui Province, China.
- **Edge-device validation.** Deployed on an NVIDIA Jetson Orin Nano 8GB via TensorRT (FP16), achieving near-real-time inference suitable for on-board UAV monitoring.

---

## 🧭 Method

HMQD-Net takes a wheat RGB image and its corresponding depth image as input and processes them through two modality-specific branches before selective fusion and query decoding.

```
                 ┌─────────────────────────┐
   RGB Image ──► │   RGB Branch  (SSHR)     │ ──► F3_r, F4_r, F5_r
                 └─────────────────────────┘             │
                                                          ▼
                                              ┌───────────────────────┐
                                              │  DTF  (P3 / P4 / P5)   │ ──► fused features
                                              │  channel align +       │
                                              │  token-selective attn   │
                                              └───────────────────────┘
                                                          ▲
                 ┌─────────────────────────┐             │
 Depth Image ──► │  Depth Branch (PDER)     │ ──► F3_d, F4_d, F5_d
                 └─────────────────────────┘
                                                          │
                                                          ▼
                                          ┌───────────────────────────┐
                                          │   RT-DETR Decoder Head      │
                                          │   + QAKD (teacher→student)  │
                                          └───────────────────────────┘
                                                          │
                                                          ▼
                                              Wheat scab detections
```

### Components

- **SSHR (RGB branch).** A series of convolutional blocks and C2f modules build multi-level RGB features, while SSBlocks are embedded into the deeper stages to enhance long-range dependency modeling. Instead of dense pixel-wise scanning, SSHR adopts a sparse process based on **region folding**, **similarity assignment**, and **feature scattering**: each feature map is folded into local regions, a small set of candidate centers is generated per region by adaptive pooling, pixels are softly assigned to their most relevant center, and a selective state-space scan models long-range context among region centers. The similarity computation scales as O(NK) (with K ≪ N) rather than O(N²).

- **PDER (depth branch).** A progressive mixture-of-experts routing mechanism embedded at the higher-level depth stages. A lightweight router selects the Top-K experts per sample, a shared expert provides a stable base representation, and an auxiliary load-balancing constraint prevents router collapse. This allocates more capacity to regions with severe occlusion, complex depth variation, or ambiguous diseased-spike boundaries.

- **DTF (fusion).** Maps dual-stream features into a unified semantic space via channel alignment, then applies **Token-selective Attention (TSA)** at the deeper levels (P4/P5), retaining only the most relevant key–value pairs per query token to suppress redundant background. Shallow features (P3) use lightweight linear fusion to preserve fine-grained lesion detail.

- **QAKD (distillation).** A homogeneous self-distillation scheme. A complete HMQD-Net is **pre-trained to convergence on the target dataset and frozen as the teacher**; a student of the same architecture is then trained, distilling only high-confidence effective queries (classification via soft-label BCE, localization via L1 + GIoU). A linear warm-up gradually introduces teacher supervision. The teacher is used only during training, so inference cost is unchanged.

---

## 📂 Dataset: WS-D

The **WS-D** dataset is a UAV-based RGB-D wheat scab detection dataset constructed through field-site selection, low-altitude multimodal image acquisition, and image preprocessing with annotation.

| Parameter | Specification |
| --- | --- |
| Collection period | March–May 2025 |
| Collection sites | Bazhen (Chaohu City), Lujiang County, Shucheng County, Anhui, China |
| Camera | Stereolabs ZED X stereo depth camera |
| Modalities | RGB + Depth |
| UAV flight altitude | 2–3 m |
| Image resolution | 640 × 640 |
| Annotation tool | Make Sense (exported in COCO format) |
| Total images | 1,950 (greenhouse 1,250 / outdoor 700) |
| Disease instances | 6,799 |
| Train / Test split | 1,560 / 390 (8:2) |

**Annotation quality control.** All diseased-spike regions were independently annotated by four agronomy experts following a unified annotation protocol. 10% of the images were randomly selected for cross-checking among the experts, and inconsistent or ambiguous annotations were resolved through joint discussion until consensus was reached, while invalid boxes were removed.

> The WS-D dataset will be released at `https://XXX` *(to be updated)* upon acceptance.

---

## ⚙️ Installation

> Code and environment files will be released upon acceptance. The intended setup is summarized below.

```bash
# 1. Clone the repository
git clone https://github.com/yangjie1874/HMQD.git
cd HMQD

# 2. Create the environment
conda create -n hmqd python=3.10 -y
conda activate hmqd

# 3. Install dependencies
pip install -r requirements.txt
# (PyTorch, torchvision, mamba-ssm, einops, pycocotools, etc.)
```

---

## 🚀 Usage

> The following commands reflect the intended released interface.

### Data preparation

```
HMQD/
├── datasets/
│   ├── WS-D/
│   │   ├── rgb/                # RGB images (640×640)
│   │   ├── depth/              # aligned depth maps
│   │   └── annotations/
│   │       ├── train.json      # COCO format
│   │       └── test.json
│   └── MSWDD/                  # pseudo-depth generated by Depth-Anything V3
```

### Training

```bash
# Stage 1: train the teacher model to convergence
python train.py --config configs/hmqd_wsd_teacher.yml

# Stage 2: train the student model with QAKD (teacher frozen)
python train.py --config configs/hmqd_wsd_student.yml --teacher weights/teacher_best.pth
```

### Evaluation

```bash
python eval.py --config configs/hmqd_wsd_student.yml --weights weights/hmqd_best.pth
```

### Inference

```bash
python infer.py --rgb path/to/rgb.png --depth path/to/depth.png --weights weights/hmqd_best.pth
```

---

## 🛠️ Implementation Details

HMQD-Net is implemented based on the RT-DETR framework, and the basic training settings follow those of the original RT-DETR to ensure a fair comparison.

| Setting | Value |
| --- | --- |
| Base framework | RT-DETR |
| Max epochs | 300 (early stopping, patience = 120) |
| Input size | 640 × 640 |
| Batch size | 4 |
| Initial learning rate (lr) | 0.01 |
| Final lr factor (lrf) | 0.01 |
| Warm-up momentum | 0.8 |
| Data augmentation | identical to RT-DETR baseline |
| Random seed | 0 (results averaged over 3 runs) |
| QAKD: query bound K | 100 |
| QAKD: confidence threshold τ | 0.05 |
| QAKD: distillation weight λ_kd | 0.5 |
| QAKD: warm-up epochs E_w | 5 |
| GPU | NVIDIA GeForce RTX 4090 |

---

## 📊 Results

### Detection results on WS-D

| Method | Modality | AP (%) | AP50 (%) | AP75 (%) | APs (%) |
| --- | --- | --- | --- | --- | --- |
| RT-DETR | RGB | 24.5 | 60.2 | 13.8 | 20.8 |
| RT-DETR-D | RGB-D | 24.4 | 60.3 | 12.9 | 20.5 |
| **HMQD-Net (Ours)** | RGB-D | **32.4** | **68.7** | **20.8** | **31.7** |

### Generalization on MSWDD

| Method | Modality | AP (%) | AP50 (%) | AP75 (%) |
| --- | --- | --- | --- | --- |
| RT-DETR-D | RGB-D | 75.6 | 95.6 | 69.3 |
| **HMQD-Net (Ours)** | RGB-D | **76.8** | **96.5** | **69.9** |

### Edge deployment

| Platform | Format | Precision | Latency (ms) | FPS | Peak Memory (MB) | AP50 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Server (RTX 4090) | PyTorch | FP32 | 11.8 | 87.4 | 3032 | 68.7 |
| Jetson Orin Nano 8GB | TensorRT | FP16 | 46.3 | 21.6 | 1925 | 65.9 |

---

## 🌾 Practical Deployment

For on-board UAV deployment, the trained model can be exported from PyTorch (`.pt`) to a TensorRT (`.engine`) format with FP16 precision, enabling near-real-time inference on resource-constrained edge hardware such as the Jetson Orin Nano. The spike-level detection results can be aggregated into field-scale infection maps to support agricultural decision-making, such as identifying high-risk regions, planning targeted fungicide spraying, and prioritizing field inspection.

```bash
# Export to ONNX, then build a TensorRT FP16 engine
python export.py --weights weights/hmqd_best.pth --format onnx
trtexec --onnx=hmqd.onnx --fp16 --saveEngine=hmqd.engine
```

---

## 📖 Citation

If you find this work useful, please consider citing our paper (full citation will be updated upon publication):

```bibtex
@article{hmqdnet,
  title   = {Hierarchical multimodal representation and query-aware distillation for UAV-based wheat scab monitoring},
  author  = {XXX and XXX and XXX},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## 🙏 Acknowledgements

This work is built upon the excellent [**RT-DETR**](https://github.com/lyuwenyu/RT-DETR) codebase. We also thank the authors of [Mamba / Vision Mamba](https://github.com/hustvl/Vim) and [Depth-Anything](https://github.com/DepthAnything) for their open-source contributions, which inspired parts of our design. We are grateful to the agronomy experts who participated in the construction and annotation of the WS-D dataset.

---

## 📬 Contact

For questions about the paper, the WS-D dataset, or the code release, please open an issue in this repository or contact the authors. We will respond and complete the public release of all code, weights, and data **upon acceptance of the paper**.

---

## 📄 License

The code and dataset will be released under an appropriate open-source / academic-use license upon publication. Until then, all rights are reserved by the authors.
