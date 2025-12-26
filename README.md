# Floorplan Denoising & Segmentation  

**AttResUNet-based denoising model** removes noise (text, dimensions, hatches) from architectural floorplans, enabling more accurate **wall/door/window segmentation** for BIM reconstruction.

---

## 1. Project Overview  
This project, developed by the **Construction AI & Robotics Team**, proposes a **Residual-Attention U-Net (AttResUNet)** for architectural drawing denoising to enhance **Mask R-CNN** instance segmentation on floorplans.  

- **Dataset:**
  - AI-Hub 2022 (48,033 images, 2.6M+ structure labels)  
  - **Type:** Apartment / Multiplex / Detached house floorplans  
  - **Class:** STR (Structure) / SPA (Space) / OBJ (Object) / OCR (Text)

- **Performance:**  
  - **Noise ROI:** PSNR 35.43, SSIM 0.9978  
  - **Structure ROI:** PSNR 24.14, SSIM 0.9852  
  - *Using class-weighted loss (background: 0.1 | wall: 0.3 | window/door: 1.0)*  

---

## 2. Key Results  

**Denoising Results:** 
| Metric | Full Image | Noise ROI | Structure ROI |
|--------|-------------|------------|----------------|
| MSE | 0.0046 | 0.0034 | 0.0608 |
| PSNR | 23.64 | 35.43 | 24.14 |
| SSIM | 0.9503 | 0.9978 | 0.9852 |
| LPIPS | 0.0629 | 0.0026 | 0.0361 |

**Segmentation Results (mAP50):**  

| Dataset | **mAP50** |
|----------|-----------|
| **Raw / Raw** | **0.2090** |
| **Raw / Clean** | 0.1032 |
| **Clean / Clean** | 0.0598 |
| **GT / Clean** | 0.1141 |

- The **Raw / Raw** configuration achieved the highest mAP50.  
- However, **GT / Clean** outperformed **Raw / Clean** and **Clean / Clean**,  
showing that well-aligned denoised images and labels could further improve segmentation quality.

---

## 3. Denoising Model Architecture  

**Residual-Attention U-Net**
- Residual blocks for stable deep feature learning and structure preservation  
- Attention gates to focus on thin structural lines  
- Hybrid encoder–decoder architecture specialized for architectural patterns  

**Loss Function**
- **BCE (0.5) + Dice + SSIM**
  - Binary cross-entropy for mask binarization  
  - Dice for overlapping region accuracy  
  - SSIM for perceptual similarity
  - *Class-weighted loss applied:* **background: 0.1 | wall: 0.3 | window/door: 1.0**
    - Class weights were applied using the **Mask dataset**, assigning higher importance to structural regions such as walls, windows, and doors.


**Training Settings**
- Dataset split: **Train 3,000 / Validation 300 / Test 300**  
- Epochs: 200 (early stopping enabled)  
- Augmentation:  
  - Flip (50%), translate ±10px, scale ±10%, rotation ±15° (70% prob.)  

---

## 4. Dataset  

- **Source:** [AI-Hub Construction Drawing Dataset (2022)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71465)  
  - **Provided Formats:** PNG images and JSON labels (structure, object, space annotations)  

- **Extended for Denoising:**  
  - Additional data created in this work:  
    - **Raw** — noisy input floorplans  
    - **GT (Ground Truth)** — clean reference drawings  
    - **Masks** — region maps for wall, window, and door  
  - These splits were derived from the AI-Hub dataset and refined for noise removal and structure segmentation tasks.

- **Characteristics of Drawing Data:**
- Binary-like high-contrast composition (**0 and 1**).  
- Extremely sensitive to pixel-level variations — even single-pixel noise significantly affects object detection and segmentation accuracy.


---

## 5. Experiments  

- **Denoising Model:**  
  - Residual-Attention U-Net
- **Segmentation Model:**
  - Mask R-CNN
 
### 5.1 Denoising Evaluation  

**Evaluation Regions**
- **Full Image:** Entire image  
- **Noise ROI:** Regions containing text, dimensions, and hatch patterns  
- **Structure ROI:** Regions with architectural elements (walls, doors, windows)
<img width="869" height="281" alt="Image" src="https://github.com/user-attachments/assets/b67abe31-7223-4c14-be8b-77c276afbde7" />

**Quantitative Results**

| Metric | Full Image | Noise ROI | Structure ROI |
|--------|-------------|------------|----------------|
| PSNR | 23.64 | 35.43 | 24.14 |
| SSIM | 0.9503 | 0.9978 | 0.9852 |
| LPIPS | 0.0629 | 0.0026 | 0.0361 |
<img width="1061" height="934" alt="Image" src="https://github.com/user-attachments/assets/bbedbec8-73aa-4a91-8066-d7384b6e398f" />

---

### 5.2 Segmentation Tests  

**Segmentation Test Variants (Train Dataset / Test Dataset):**
1. Raw / Raw  
2. Raw / Clean  
3. Clean / Clean  
4. GT / Clean
 
| Dataset | **mAP50** |
|----------|-----------|
| **Raw / Raw** | **0.2090** |
| **Raw / Clean** | 0.1032 |
| **Clean / Clean** | 0.0598 |
| **GT / Clean** | 0.1141 |

> Tests (2) and (3) were designed with the expectation that denoised images would yield higher segmentation performance.  
> However, (1) **Raw / Raw** achieved the highest mAP50, while (2) and (3) underperformed due to label alignment with **raw images**, causing mismatches on clean inputs.  
> The (4) **GT / Clean** test showed noticeable improvement over (2) and (3), indicating that with properly aligned **denoised images and labels**, segmentation performance could be further enhanced.
<img width="243" height="488" alt="Image" src="https://github.com/user-attachments/assets/ccbb4361-5d00-4e4c-929c-77018306fc79" />

**Insights:**  
- Structure restoration > generic noise removal  
- Proper label–image alignment is critical for segmentation performance  
- Future directions : line continuity and junction-aware loss functions  

---

## 6. Environment  

Trained on **Mac M4** and **Google Colab** environments.  

---

## 7. Future Work  

- Structure-aware denoising focused on line continuity and geometric accuracy  
- Junction-aware loss for boundary refinement  
- Integration into a real-time BIM reconstruction pipeline
