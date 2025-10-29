# 🧠 Retinal Vessel Segmentation using the DRIVE Dataset  

### Biomedical Image Processing & Deep Learning Project  
**Author:** Yosra Said  

---
## 🌍 Project Overview  

This project focuses on **automatic segmentation of retinal blood vessels** from fundus images using **deep learning**.  
Segmenting retinal vessels plays a crucial role in early diagnosis of diseases such as **diabetic retinopathy**, **glaucoma**, and **hypertension**.  

**The goals are to:**  
1. Preprocess the raw **DRIVE** and **CHASE_DB1** datasets to standardize illumination and size.  
2. Train a deep learning model (such as **U-Net** or **DeepLabV3+**) for binary vessel segmentation.  
3. **Evaluate model performance both within and across datasets** to study cross-dataset generalization.  

Specifically, the model will be trained and tested under four configurations to assess **cross-correlation** between the two datasets:  
- 🧠 **Train on DRIVE → Test on DRIVE** (baseline performance)  
- 🧠 **Train on CHASE_DB1 → Test on CHASE_DB1** (baseline performance)  
- 🔄 **Train on DRIVE → Test on CHASE_DB1** (cross-dataset generalization)  
- 🔄 **Train on CHASE_DB1 → Test on DRIVE** (cross-dataset generalization)  

Evaluation metrics include the **Dice coefficient**, **Intersection over Union (IoU)**, **Sensitivity**, and **Specificity**.  

This repository includes preprocessing, data management, and will later include full model training and evaluation scripts for these experiments.  

---

## 📊 Dataset Description  

### 🩺 DRIVE — *Digital Retinal Images for Vessel Extraction*  

The **DRIVE** dataset is a standard benchmark for retinal vessel segmentation.  

| Property | Description |
|-----------|-------------|
| **Images** | 40 color fundus photographs (20 training + 20 test) |
| **Resolution** | 565 × 584 pixels |
| **Annotations** | Manual vessel masks provided by experts |
| **Task** | Binary segmentation: vessel (1) vs. background (0) |
| **Source** | [DRIVE Challenge Dataset](https://drive.grand-challenge.org/) |

**Structure:**  
- `images/` — RGB fundus images  
- `1st_manual/` — ground truth vessel masks  
- `mask/` — field-of-view (FOV) masks  

Each image is accompanied by a binary mask highlighting blood vessels within the FOV.  

---

### 🧬 CHASE_DB1 — *Child Heart and Health Study in England*  

The **CHASE_DB1** dataset contains high-resolution fundus images with manually annotated vessel segmentations by two observers.  

| Property | Description |
|-----------|-------------|
| **Images** | 28 color fundus photographs (14 subjects × 2 eyes) |
| **Resolution** | 999 × 960 pixels |
| **Annotations** | Two manual vessel annotations per image |
| **Task** | Binary segmentation: vessel (1) vs. background (0) |
| **Source** | [CHASE_DB1 Dataset – University of Kingston](https://blogs.kingston.ac.uk/retinal/chasedb1/) |

---

## ⚙️ Preprocessing Pipeline  

All preprocessing is implemented in [`src/preprocess_all.py`](src/preprocess_all.py).  

**The script performs the following operations:**  
1. **Load & Inspect Data** – reads all training images and masks  
2. **RGB Conversion** – converts images from BGR (OpenCV) to RGB format  
3. **Normalization** – scales intensity values to [0, 1]  
4. **FOV Masking** – removes black background outside the retinal area  
5. **Resizing** – resizes all images and masks to 512×512 pixels  
6. **Saving Results** – saves preprocessed images to `data/preprocessed_images/`  

---

## 🧩 Planned Deep Learning Model  

The next stage of this project will implement a **U-Net-based segmentation model** in PyTorch.  

**Model goals:**  
- **Input:** preprocessed RGB fundus images  
- **Output:** binary vessel segmentation masks  
- **Loss functions:** Dice Loss, Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Evaluation metrics:** Dice, IoU, Sensitivity, Specificity  

**Future extensions will include:**  
- Transfer learning with pre-trained encoders (e.g. ResNet-based U-Net)  
- Confidence maps for uncertainty quantification  
- Model explainability using Grad-CAM  
- **Cross-dataset analysis** to quantify the generalization gap between DRIVE and CHASE_DB1 and understand inter-dataset correlations.  

---

## 🧪 Planned Experiments  

| Experiment | Train Dataset | Test Dataset | Purpose |
|-------------|----------------|---------------|----------|
| **E1** | DRIVE | DRIVE | Baseline performance on DRIVE |
| **E2** | CHASE_DB1 | CHASE_DB1 | Baseline performance on CHASE_DB1 |
| **E3** | DRIVE | CHASE_DB1 | Test generalization from DRIVE → CHASE |
| **E4** | CHASE_DB1 | DRIVE | Test generalization from CHASE → DRIVE |


---

## ✨ Acknowledgment  

The **DRIVE dataset** is provided by the Image Sciences Institute, University Medical Center Utrecht.  
**Original publication:**  
> J. Staal et al., *"Ridge-based vessel segmentation in color images of the retina,"* IEEE Transactions on Medical Imaging, 2004.  

---

## 🧑‍💻 Author  

**Yosra Said**  
Biomedical Engineer • Deep Learning for Medical Imaging  
📍 Telecom Physique Strasbourg – Politecnico di Milano Exchange  
📧 [yosrasaid01@gmail.com](mailto:yosrasaid01@gmail.com)  
🌐 [LinkedIn Profile](https://www.linkedin.com/in/yosra-said-925131257/)
