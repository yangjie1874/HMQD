# HMQD-Net: Hierarchical Multimodal Representation and Query-Aware Distillation for UAV-Based Wheat Scab Detection

> Official repository for the paper **"Hierarchical multimodal representation and query-aware distillation for UAV-based wheat scab monitoring"**, submitted to *Engineering Applications of Artificial Intelligence (EAAI)*.

---

## 📌 Overview

**HMQD-Net** is a hierarchical multimodal representation and query-aware distillation network for **RGB-D wheat scab detection** from unmanned aerial vehicle (UAV) imagery. UAV remote sensing provides a feasible solution for large-scale wheat disease monitoring, but wheat scab detection remains challenging because diseased spikes are small, partially occluded by dense canopy structures, and difficult to distinguish using single-modal images alone.

HMQD-Net is built on top of [**RT-DETR**](https://github.com/lyuwenyu/RT-DETR) and integrates RGB appearance cues with depth structural cues through modality-specific representation learning, selective cross-modal fusion, and decoder-level query refinement.

The framework contains four key designs:

- **SSHR** — Sparse State-space Hierarchical Representation for the RGB branch.
- **PDER** — Progressive Depth Expert Routing for the depth branch.
- **DTF** — Dual-Stream Token-selective Fusion for RGB-D feature interaction.
- **QAKD** — Query-Aware Knowledge Distillation for decoder-level query refinement.

HMQD-Net achieves **32.4% AP** and **68.7% AP50** on the self-constructed **WS-D** dataset, and achieves **76.8% AP** on the public **MSWDD** dataset with pseudo-depth inputs, showing its applicability to UAV-based wheat disease monitoring.

---

## 🔔 Release Status

> **Important.** This repository is being prepared for public release. The full source code, fixed split files, preprocessing scripts, training configurations, pretrained weights, and the WS-D dataset will be made publicly available upon paper acceptance.

| Resource | Status | Link |
| --- | --- | --- |
| Method documentation | ✅ Available | This README |
| Source code | 🔒 Upon acceptance | `TODO_UPDATE` |
| WS-D dataset | 🔒 Upon acceptance | `TODO_UPDATE` |
| Fixed split files | 🔒 Upon acceptance | `splits/wsd/*.txt` |
| Pretrained weights | 🔒 Upon acceptance | `weights/*.pt` |
| Reproduction scripts | 🔒 Upon acceptance | `scripts/*.sh` |

Temporary dataset/model link used during review:

```text
https://drive.google.com/drive/folders/1eFtnyeHkhj9uz_DNVlhjlGOtXyIiygN8?usp=drive_link
```

---

## ✨ Highlights

- **UAV RGB-D wheat scab detection.** HMQD-Net explicitly combines RGB appearance information and depth geometric information for disease target detection in complex field scenes.
- **Modality-specific feature encoding.** SSHR enhances long-range contextual modeling of disease appearance, while PDER captures disease-relevant depth structures through progressive expert routing.
- **Selective multimodal fusion.** DTF performs multi-scale channel alignment and token-selective attention to suppress redundant background interactions.
- **Query-aware distillation.** QAKD distills only high-confidence effective queries from a frozen teacher model, improving query-level prediction without increasing inference cost.
- **Dataset contribution.** WS-D contains 1,950 spatially aligned RGB-D image pairs and 6,799 annotated diseased-spike instances.

---

## 🧭 Method Overview

HMQD-Net takes spatially aligned RGB and depth images as input. The RGB image is processed by the SSHR-enhanced RGB branch, while the depth image is processed by the PDER-enhanced depth branch. Multi-scale features from both branches are then fused by DTF at P3, P4, and P5. The fused features are finally decoded by an RT-DETR decoder head. During training, QAKD uses a frozen teacher model to guide the student by distilling only high-confidence queries.

```text
RGB image   ──► RGB branch with SSHR   ──► RGB features ┐
                                                        ├─► DTF ─► RT-DETR decoder ─► detections
Depth image ──► Depth branch with PDER ──► Depth features┘

Teacher decoder outputs ──► high-confidence query selection ──► QAKD ──► student training
```

### Main components

- **SSHR.** SSBlocks are embedded into deeper RGB stages to perform sparse region-level state-space scanning. Instead of scanning all pixels densely, SSHR uses region folding, similarity-based assignment, sparse state-space scanning, and feature scattering to capture long-range disease-context dependencies with reduced redundancy.
- **PDER.** DERBlocks are inserted into deeper depth stages. A lightweight router selects Top-K experts, a shared expert stabilizes representation learning, and an auxiliary load-balancing loss prevents expert collapse.
- **DTF.** RGB and depth features are first channel-aligned. P3 uses lightweight fusion to retain fine-grained disease appearance, while P4 and P5 use token-selective attention to retain the most relevant cross-modal token interactions.
- **QAKD.** A pretrained HMQD-Net teacher is frozen. For each image, teacher queries with confidence above `0.05` are retained, and at most `K=100` queries are distilled. Classification distillation uses soft-label BCE, and box distillation uses L1 + GIoU.

---

## 📂 Repository Structure

The released repository will follow the structure below:

```text
HMQD/
├── ultralytics/                         # model, training, validation, inference code
├── configs/
│   ├── hmqd_wsd_teacher.yaml
│   ├── hmqd_wsd_student_qakd.yaml
│   ├── hmqd_mswdd.yaml
│   └── ablation/
├── datasets/
│   ├── WS-D/
│   │   ├── rgb/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── depth/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── annotations/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   └── MSWDD/
│       ├── rgb/
│       ├── pseudo_depth_depthv3/
│       └── annotations/
├── splits/
│   ├── wsd_train.txt
│   ├── wsd_val.txt
│   ├── wsd_test.txt
│   ├── mswdd_train.txt
│   ├── mswdd_val.txt
│   └── mswdd_test.txt
├── scripts/
│   ├── preprocess_wsd_depth.py
│   ├── generate_mswdd_depthv3.py
│   ├── train_teacher.sh
│   ├── train_student_qakd.sh
│   ├── eval_wsd.sh
│   ├── eval_mswdd.sh
│   └── export_tensorrt.sh
├── weights/
├── tools/
├── environment.yaml
├── requirements.txt
└── README.md
```

---

## 📂 Dataset: WS-D

The **WS-D** dataset is a UAV-based RGB-D wheat scab detection dataset constructed by low-altitude RGB-D image acquisition, RGB-depth registration, cropping, annotation, and fixed train/validation/test partitioning.

### WS-D acquisition summary

| Item | Specification |
| --- | --- |
| Collection period | March–May 2025 |
| Collection sites | Bazhen, Lujiang, Shucheng, Anhui Province, China |
| Bazhen coordinate | 31.7175334°N, 117.4275124°E; altitude 7.48 m |
| Lujiang coordinate | 31.8740037°N, 116.6825007°E; altitude 35.17 m |
| Shucheng coordinate | 31.4486161°N, 116.9782291°E; altitude 17.57 m |
| Camera | Stereolabs ZED X stereo RGB-D camera |
| Modalities | RGB + depth |
| UAV flight altitude | 2–3 m above canopy |
| Registered image size | 640 × 640 pixels |
| Annotation tool | Make Sense |
| Annotation format | COCO detection format |
| Total source images | 1,950 |
| Greenhouse / outdoor images | 1,250 / 700 |
| Annotated diseased-spike instances | 6,799 |

### Fixed WS-D split

The dataset was split **at the source-image level before any cropping or augmentation**. The 1,950 source images were first divided into a training pool and a test set at an 8:2 ratio. The training pool was then split into training and validation subsets at an 8:2 ratio. The final train/validation/test proportions are therefore 64%/16%/20%.

| Subset | Images | Instances | Proportion |
| --- | ---: | ---: | ---: |
| Train | 1,248 | 4,620 | 64% |
| Validation | 312 | 1,029 | 16% |
| Test | 390 | 1,150 | 20% |
| Total | 1,950 | 6,799 | 100% |

### COCO-scale distribution

| Scale | COCO definition | Count | Proportion |
| --- | --- | ---: | ---: |
| Small | area < 32² pixels | 3,529 | 51.9% |
| Medium | 32² ≤ area < 96² pixels | 3,197 | 47.0% |
| Large | area ≥ 96² pixels | 73 | 1.1% |

### Field/session metadata

The following metadata files will be released with the dataset to help users check potential spatial or acquisition-level dependence:

```text
metadata/wsd_image_metadata.csv
metadata/wsd_site_plot_flight_session_summary.csv
```

The metadata table will include at least the following fields:

| Field | Description |
| --- | --- |
| `image_id` | Unique source image identifier |
| `subset` | train / val / test |
| `site` | Bazhen / Lujiang / Shucheng |
| `plot_id` | Field plot identifier |
| `flight_id` | UAV flight identifier |
| `session_id` | Acquisition session identifier |
| `capture_mode` | Stop-and-shoot single-frame capture |
| `rgb_path` | RGB file path |
| `depth_path` | Depth file path |
| `annotation_path` | COCO annotation file |

> **TODO_VERIFY before final public release:** update exact site/plot/flight/session counts in `metadata/wsd_site_plot_flight_session_summary.csv` after final metadata audit.

### Annotation reliability

All images were annotated by agronomy-trained annotators using a unified protocol. Each source image was assigned to one primary annotator. A randomly selected 10% subset was independently cross-checked by a second annotator. Disagreements were resolved by consensus discussion.

The released repository will include:

```text
metadata/annotation_protocol.md
metadata/inter_annotator_agreement.csv
metadata/cross_check_discrepancies.csv
```

The cross-check table will report:

| Metric | Value |
| --- | ---: |
| Cross-checked images | 195 |
| IoU threshold for box agreement | 0.50 |
| Matched boxes | TODO_VERIFY |
| Discrepant boxes | TODO_VERIFY |
| Corrected boxes | TODO_VERIFY |
| Removed boxes | TODO_VERIFY |
| Added boxes after consensus | TODO_VERIFY |
| Box-level agreement rate | TODO_VERIFY |

> The `TODO_VERIFY` values are placeholders for the final audited annotation-quality statistics and should be replaced before repository release.

---

## 🌊 Depth Preprocessing

### WS-D real depth

Depth maps in WS-D were acquired using the Stereolabs ZED X camera and spatially aligned to RGB images using the manufacturer-provided SDK. The depth values are handled as metric depth and converted to meters in preprocessing.

Invalid depth pixels are detected using the following rules:

- non-finite values: `NaN` or `Inf`;
- zero or negative depth values;
- values outside the practical valid range used in this study;
- isolated invalid pixels after registration.

Because the UAV flight height was 2–3 m and the camera operated within its effective sensing range, the practical valid range was set to:

```text
valid_depth_range = [0.3 m, 20.0 m]
```

The preprocessing pipeline is:

1. RGB-depth registration using the Stereolabs SDK.
2. Invalid-value masking according to the rules above.
3. Local filtering of isolated invalid pixels.
4. Hole filling by nearest-neighbor interpolation from surrounding valid pixels.
5. Light median filtering to suppress local noise while preserving spike boundaries.
6. Depth clipping to `[0.3 m, 20.0 m]`.
7. Min–max normalization to `[0, 1]`.
8. Encoding as a single-channel floating-point depth image.

The corresponding implementation will be released as:

```bash
python scripts/preprocess_wsd_depth.py \
  --rgb-dir datasets/WS-D/raw_rgb \
  --depth-dir datasets/WS-D/raw_depth \
  --out-dir datasets/WS-D/depth \
  --min-depth 0.3 \
  --max-depth 20.0 \
  --fill nearest \
  --median-kernel 3
```

### MSWDD pseudo-depth

MSWDD does not provide real depth measurements. Therefore, pseudo-depth maps were generated from RGB images using **Depth Anything V3**. These maps do not represent metric sensor depth and are used only as relative structural cues.

```bash
python scripts/generate_mswdd_depthv3.py \
  --rgb-root datasets/MSWDD/rgb \
  --out-root datasets/MSWDD/pseudo_depth_depthv3 \
  --model depth-anything-v3 \
  --normalize minmax
```

Pseudo-depth maps are generated independently for the training, validation, and test partitions. Test-set pseudo-depth maps are used only during evaluation and are never used for training or model selection.

---

## 🔁 Data Augmentation

All geometric transformations are applied **synchronously** to RGB images, depth maps, and bounding boxes using identical transformation parameters to preserve cross-modal spatial consistency.

The released training configuration will include the complete augmentation settings. The default settings used for the reported experiments are summarized below.

| Operation | Probability | Parameter range / setting | Applied to |
| --- | ---: | --- | --- |
| Horizontal flip | 0.5 | left-right flip | RGB, depth, boxes |
| Random scale | 1.0 | 0.5–1.5 | RGB, depth, boxes |
| Random translation | 1.0 | ±10% image size | RGB, depth, boxes |
| Random crop / resize | 1.0 | output 640 × 640 | RGB, depth, boxes |
| HSV color jitter | 1.0 | h=0.015, s=0.7, v=0.4 | RGB only |
| Mosaic | 0.0 | disabled for RGB-D alignment stability | — |
| MixUp | 0.0 | disabled | — |

> **TODO_VERIFY before final release:** ensure these values exactly match the final YAML training configuration. If the final RT-DETR baseline uses different augmentation ranges, update this table and `configs/*.yaml` consistently.

---

## ⚙️ Installation

```bash
# 1. Clone repository
git clone https://github.com/yangjie1874/HMQD.git
cd HMQD

# 2. Create environment
conda create -n hmqd python=3.10 -y
conda activate hmqd

# 3. Install PyTorch, then project dependencies
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Install optional acceleration packages
pip install mamba-ssm==1.2.0.post1 causal-conv1d==1.2.0.post2
```

### Tested environment

| Item | Version |
| --- | --- |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.1.2 |
| Torchvision | 0.16.2 |
| CUDA | 12.1 |
| cuDNN | 8.9 |
| Ultralytics base | TODO_VERIFY commit hash |
| mamba-ssm | 1.2.0.post1 |
| einops | 0.7.0 |
| pycocotools | 2.0.7 |
| OpenCV | 4.8.1 |
| GPU for training | NVIDIA GeForce RTX 4090 |
| Edge device | NVIDIA Jetson Orin Nano 8GB |
| JetPack | 6.1 |
| TensorRT | 10.3 |

---

## 🛠️ Implementation Details

| Setting | Value |
| --- | --- |
| Base detector | RT-DETR |
| Input size | 640 × 640 |
| Batch size | 4 |
| Maximum epochs | 300 |
| Early-stopping patience | 120 epochs |
| Checkpoint selection | best validation AP / validation fitness (`TODO_VERIFY`) |
| Optimizer | SGD (`TODO_VERIFY`) |
| Initial learning rate | 0.01 |
| Final learning-rate factor | 0.01 |
| Weight decay | 0.0005 (`TODO_VERIFY`) |
| Momentum | 0.937 (`TODO_VERIFY`) |
| Warm-up momentum | 0.8 |
| LR schedule | cosine schedule (`TODO_VERIFY`) |
| Pretrained initialization | RT-DETR/ImageNet-pretrained backbone (`TODO_VERIFY`) |
| Decoder queries | 300 |
| Confidence threshold for evaluation | 0.001 (`TODO_VERIFY`) |
| NMS | not used for RT-DETR-style set prediction |
| Evaluation metrics | COCO AP, AP50, AP75, APs, APm, APl |
| Random seeds for repeated runs | 0, 49, 99 |

### Module-specific settings

| Component | Setting |
| --- | --- |
| SSHR insertion | P4 and P5 RGB stages |
| SSBlock repeats | P4=1, P5=3 |
| PDER insertion | P4 and P5 depth stages |
| PDER experts | P4=4, P5=8 (`P3=4 if enabled`) |
| PDER Top-K experts | 2 |
| DTF attention heads | 8 |
| DTF token selection ratio | 0.8 |
| DTF channel groups | 4 |
| QAKD query confidence threshold | 0.05 |
| QAKD maximum selected queries | 100 |
| QAKD classification weight | 1.0 |
| QAKD box weight | 1.0 |
| QAKD overall weight | 0.5 |
| QAKD warm-up epochs | 5 |

---

## 🚀 Training and Evaluation

### Stage 1: train teacher model

```bash
bash scripts/train_teacher.sh
```

Equivalent command:

```bash
python train.py \
  --data configs/data/wsd.yaml \
  --model configs/hmqd_wsd_teacher.yaml \
  --imgsz 640 \
  --batch 4 \
  --epochs 300 \
  --patience 120 \
  --seed 0 \
  --project runs/wsd \
  --name hmqd_teacher
```

### Stage 2: train student model with QAKD

```bash
bash scripts/train_student_qakd.sh
```

Equivalent command:

```bash
python train.py \
  --data configs/data/wsd.yaml \
  --model configs/hmqd_wsd_student.yaml \
  --teacher weights/hmqd_teacher_best.pt \
  --distill qakd \
  --qakd-conf 0.05 \
  --qakd-topk 100 \
  --qakd-weight 0.5 \
  --qakd-warmup 5 \
  --imgsz 640 \
  --batch 4 \
  --epochs 300 \
  --patience 120 \
  --seed 0 \
  --project runs/wsd \
  --name hmqd_student_qakd
```

### Evaluation on WS-D

```bash
bash scripts/eval_wsd.sh
```

```bash
python val.py \
  --data configs/data/wsd.yaml \
  --weights weights/hmqd_best.pt \
  --imgsz 640 \
  --batch 1 \
  --task test
```

### Evaluation on MSWDD

```bash
bash scripts/eval_mswdd.sh
```

```bash
python train.py \
  --data configs/data/mswdd.yaml \
  --model configs/hmqd_mswdd.yaml \
  --imgsz 640 \
  --batch 4 \
  --epochs 300 \
  --seed 0

python val.py \
  --data configs/data/mswdd.yaml \
  --weights weights/hmqd_mswdd_best.pt \
  --imgsz 640 \
  --task test
```

---

## 📊 Principal Results

### WS-D test set

Results are reported as mean ± standard deviation over three independent runs using fixed data partitions and seeds 0, 49, and 99 where available.

| Method | Modality | AP (%) | AP50 (%) | AP75 (%) | APs (%) | APm (%) | APl (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN | RGB | 11.4 | 40.2 | 9.8 | 15.5 | 25.9 | 32.1 |
| YOLOv8s | RGB | 24.8 | 57.7 | 12.1 | 20.1 | 26.3 | 37.5 |
| YOLOv11s | RGB | 23.2 | 58.1 | 12.6 | 19.4 | 27.2 | 36.7 |
| YOLOv26s | RGB | 24.0 | 58.6 | 12.3 | 19.3 | 26.4 | 38.3 |
| D-FINE | RGB | 23.6 | 59.2 | 12.9 | 20.4 | 27.1 | 38.1 |
| DEIM | RGB | 24.1 | 59.8 | 12.4 | 21.7 | 28.3 | 38.4 |
| RT-DETR | RGB | 24.5±0.2 | 60.2±0.1 | 13.8±0.3 | 20.8±0.3 | 28.7±0.3 | 37.9±0.2 |
| RT-DETR-D | RGB-D | 24.4±0.2 | 60.3±0.2 | 12.9±0.1 | 20.5±0.1 | 28.4±0.3 | 38.1±0.1 |
| MEDUSA | RGB-D | 15.2±0.5 | 46.8±0.4 | 10.3±0.7 | 19.7±0.4 | 25.0±0.5 | 33.9±0.4 |
| ICAFusion | RGB-D | 23.0±0.3 | 56.9±0.3 | 12.4±0.2 | 18.9±0.3 | 26.9±0.4 | 38.3±0.4 |
| **HMQD-Net** | RGB-D | **32.4±0.1** | **68.7±0.1** | **20.8±0.2** | **31.7±0.1** | **36.9±0.2** | **45.5±0.2** |

### Module ablation on WS-D validation set

| SSHR | PDER | DTF | QAKD | AP (%) | AP50 (%) | AP75 (%) | APs (%) | APm (%) | APl (%) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  | 24.4±0.2 | 60.3±0.2 | 12.9±0.1 | 20.5±0.1 | 28.4±0.3 | 38.1±0.1 |
| ✓ |  |  |  | 26.2±0.1 | 65.8±0.2 | 13.3±0.2 | 26.1±0.1 | 30.1±0.1 | 40.2±0.2 |
|  | ✓ |  |  | 25.1±0.3 | 64.0±0.2 | 12.8±0.1 | 25.3±0.2 | 29.4±0.3 | 39.9±0.2 |
|  |  | ✓ |  | 24.9±0.2 | 62.0±0.1 | 13.3±0.1 | 22.4±0.2 | 29.1±0.3 | 39.4±0.3 |
|  |  |  | ✓ | 25.0±0.1 | 61.4±0.1 | 15.9±0.2 | 23.7±0.1 | 29.3±0.2 | 40.0±0.2 |
| ✓ | ✓ |  |  | 27.5±0.3 | 66.0±0.2 | 14.8±0.3 | 27.8±0.3 | 31.8±0.1 | 42.3±0.2 |
| ✓ | ✓ | ✓ |  | 32.3±0.1 | 68.4±0.1 | 20.5±0.2 | 31.5±0.1 | 36.5±0.2 | 45.2±0.1 |
| ✓ | ✓ | ✓ | ✓ | **32.4±0.1** | **68.7±0.3** | **20.8±0.3** | **31.7±0.2** | **36.9±0.2** | **45.5±0.1** |

### QAKD training overhead

QAKD is a training-time refinement. It uses a frozen teacher forward pass during student training but does not modify the student architecture used for inference.

| Training setting | GPU-hours | Peak training memory | Inference overhead |
| --- | ---: | ---: | --- |
| HMQD-Net without QAKD | 3.2 | 4,421 MB | none |
| HMQD-Net with QAKD | 5.3 | 5,623 MB | none |

> The numbers above describe the student training stage. Teacher pretraining cost should be counted separately when reporting total training budget. `TODO_VERIFY`: update total teacher + student GPU-hours after final timing audit.

### MSWDD external dataset validation

All methods are trained from scratch on the official MSWDD training split and evaluated on the corresponding test split. Since MSWDD does not contain depth images, pseudo-depth maps are generated using Depth Anything V3.

| Method | Modality | AP (%) | AP50 (%) | AP75 (%) | APs (%) | APm (%) | APl (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faster R-CNN | RGB | 56.1 | 85.4 | 42.5 | — | 64.2 | 70.9 |
| YOLOv8s | RGB | 72.1 | 94.1 | 68.9 | — | 76.3 | 80.4 |
| YOLOv11s | RGB | 72.5 | 94.3 | 68.4 | — | 76.2 | 80.5 |
| YOLOv26s | RGB | 71.6 | 93.7 | 68.0 | — | 75.8 | 80.1 |
| D-FINE | RGB | 73.8 | 95.2 | 69.2 | — | 77.2 | 81.0 |
| DEIM | RGB | 74.1 | 95.8 | 69.4 | — | 77.6 | 81.3 |
| RT-DETR | RGB | 74.2 | 95.5 | 69.1 | — | 78.0 | 80.2 |
| RT-DETR-D | RGB-D | 75.6 | 95.6 | 69.3 | — | 79.4 | 85.6 |
| **HMQD-Net** | RGB-D | **76.8** | **96.5** | **69.9** | — | **81.7** | **90.5** |

> `APs` is not reported because MSWDD does not contain small objects under the COCO scale definition used in our evaluation.

---

## 🚀 Edge Deployment

The trained model can be exported from PyTorch to TensorRT FP16 for edge-device evaluation.

```bash
python export.py --weights weights/hmqd_best.pt --format onnx --imgsz 640
trtexec --onnx=hmqd.onnx --fp16 --saveEngine=hmqd_fp16.engine
```

The reported Jetson latency includes preprocessing, model inference, and post-processing at batch size 1. Measurements use 50 warm-up runs followed by 100 timed repetitions.

| Platform | Runtime | Precision | Power mode | Latency | FPS | Peak memory | AP50 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| RTX 4090 | PyTorch | FP32 | — | 11.8 ms | 84.7 | 3,032 MB | 68.7 |
| Jetson Orin Nano 8GB | TensorRT 10.3 | FP16 | default | 46.3 ms | 21.6 | 1,925 MB | 65.9 |

> The Jetson experiment evaluates feasibility for future onboard deployment. It does not include full UAV-flight validation, camera acquisition latency, power consumption under flight conditions, thermal throttling analysis, or payload/flight-duration constraints.

---

## 📌 Reproducibility Checklist

Before public release, the repository will include:

- [ ] complete WS-D RGB-D dataset or documented access procedure;
- [ ] fixed train/validation/test split files;
- [ ] COCO annotations for all subsets;
- [ ] source-image metadata with site/plot/flight/session IDs;
- [ ] RGB-depth preprocessing scripts;
- [ ] MSWDD pseudo-depth generation scripts;
- [ ] complete training configuration files;
- [ ] augmentation settings with probabilities and ranges;
- [ ] pretrained weights for principal models;
- [ ] ablation model configurations;
- [ ] environment files and tested library versions;
- [ ] commands to reproduce each principal table;
- [ ] annotation protocol and inter-annotator agreement files;
- [ ] TensorRT export and edge-evaluation scripts.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{hmqdnet2026,
  title   = {Hierarchical multimodal representation and query-aware distillation for UAV-based wheat scab monitoring},
  author  = {TODO_UPDATE},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## 🙏 Acknowledgements

This work is built upon the excellent [RT-DETR](https://github.com/lyuwenyu/RT-DETR) codebase. We also thank the authors of Mamba, Depth Anything, MEDUSA, and ICAFusion for their open-source contributions and related studies.

---

## 📬 Contact

For questions about the paper, dataset, code, or reproduction protocol, please open an issue or contact the authors. Code, data, fixed splits, pretrained weights, and reproduction scripts will be released upon paper acceptance.

---

## 📄 License

The code and dataset will be released under an appropriate academic-use/open-source license upon publication. Until then, all rights are reserved by the authors.
