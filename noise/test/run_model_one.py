import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from noise.train.res_att_unet import AttResUNet
from utils import load_model, plot_metrics, generate_rois
import torch.nn.functional as F

# 1. Config
model_path = "../model/denoising.pth"
test_image_path = "../01-1.정식개방데이터/Test/01.원천데이터/VS_STR/APT_FP_STR_091202334.PNG"
gt_image_path = "../01-1.정식개방데이터/Test/00.클린데이터/VL_STR/APT_FP_STR_091202334.png"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_metrics_roi(preds, targets, roi_mask=None, data_range=1.0, device="cpu"):
    """
    preds, targets: [1,1,H,W], float, [0,1]
    roi_mask: [H,W] 또는 None (None이면 전체 이미지)
    """
    preds = preds.to(device)
    targets = targets.to(device)

    if roi_mask is not None:
        if isinstance(roi_mask, np.ndarray):
            roi_mask = torch.from_numpy(roi_mask)
        if roi_mask.dim() == 2:
            roi_mask = roi_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif roi_mask.dim() == 3:
            roi_mask = roi_mask.unsqueeze(1)
        roi_mask = roi_mask.to(device).float()

        preds_roi = preds * roi_mask
        targets_roi = targets * roi_mask

        # MSE: ROI 안 픽셀만 평균
        valid = roi_mask.sum() * preds.shape[1] + 1e-8
        mse = ((preds_roi - targets_roi) ** 2).sum() / valid
    else:
        preds_roi = preds
        targets_roi = targets
        mse = F.mse_loss(preds_roi, targets_roi, reduction="mean")

    # PSNR, SSIM
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    psnr = psnr_metric(preds_roi, targets_roi)
    ssim = ssim_metric(preds_roi, targets_roi)

    return {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
    }


# 2. Load model
model = AttResUNet(in_channels=1, out_channels=1)
model = load_model(model, model_path, device)


# 3. Load images
input_img = Image.open(test_image_path).convert("L")
gt_img = Image.open(gt_image_path).convert("L")
original_size = input_img.size


# 4. Preprocessing (512x512)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
input_tensor = transform(input_img).unsqueeze(0).to(device)  # [1,1,512,512]

gt_img_resized = gt_img.resize((512, 512), resample=Image.BILINEAR)
gt_img_np = np.array(gt_img_resized)  # [512,512] uint8


# 5. Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    clean_image = output[0, 0].cpu().numpy()  # [H,W] float [0,1]
    clean_image_uint8 = (clean_image * 255).astype(np.uint8)


# 6. Sharpening
sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_img = cv2.filter2D(clean_image_uint8, -1, sharp_kernel)


# 7. Prepare tensors for metrics [1,1,512,512] float [0,1]
orig_img_np = np.array(input_img.resize((512, 512), resample=Image.BILINEAR))
pred_resized_np = np.array(Image.fromarray(sharpened_img).resize((512, 512), resample=Image.BILINEAR))
preds = torch.from_numpy(pred_resized_np / 255.0).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
targets = torch.from_numpy(gt_img_np / 255.0).float().unsqueeze(0).unsqueeze(0)     # [1,1,H,W]


# 8. Generate ROIs & Compute ALL metrics
noise_roi, struct_roi = generate_rois(orig_img_np, gt_img_np)


# 전체 이미지
metrics_full = compute_metrics_roi(preds, targets, roi_mask=None, device=device)
# Noise ROI
metrics_noise = compute_metrics_roi(preds, targets, roi_mask=noise_roi, device=device)
# Structure ROI  
metrics_struct = compute_metrics_roi(preds, targets, roi_mask=struct_roi, device=device)


# 9. Print results
print("=" * 50)
print("🧹 DENOISING MODEL EVALUATION RESULTS")
print("=" * 50)
print("📊 [Full Image]")
print(f"  MSE: {metrics_full['mse']:.4f} | PSNR: {metrics_full['psnr']:.2f}dB | SSIM: {metrics_full['ssim']:.4f}")
print("🔇 [Noise ROI - 노이즈 제거 성능]")
print(f"  MSE: {metrics_noise['mse']:.4f} | PSNR: {metrics_noise['psnr']:.2f}dB | SSIM: {metrics_noise['ssim']:.4f}")
print("🏗️  [Structure ROI - 구조 보존 성능]") 
print(f"  MSE: {metrics_struct['mse']:.4f} | PSNR: {metrics_struct['psnr']:.2f}dB | SSIM: {metrics_struct['ssim']:.4f}")
print("=" * 50)


# 10. Visualize ROIs & Images
plt.figure(figsize=(20, 12))

plt.subplot(3, 4, 1); plt.title("Input (Noisy)"); plt.imshow(orig_img_np, cmap='gray')
plt.subplot(3, 4, 2); plt.title("Prediction (Denoised)"); plt.imshow(pred_resized_np, cmap='gray')
plt.subplot(3, 4, 3); plt.title("GT (Clean)"); plt.imshow(gt_img_np, cmap='gray')
plt.subplot(3, 4, 4); plt.title("Noise ROI"); plt.imshow(noise_roi, cmap='gray')
plt.subplot(3, 4, 5); plt.title("Structure ROI"); plt.imshow(struct_roi, cmap='gray')

# ✅ Metrics bar plot
regions = ['Full Image', 'Noise ROI', 'Structure ROI']
x = np.arange(len(regions))
width = 0.6

# MSE 비교
plt.subplot(3, 4, 9); plt.title("MSE Comparison")
plt.bar(x, [metrics_full['mse'], metrics_noise['mse'], metrics_struct['mse']], 
        width=width, color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('MSE')
plt.xticks(x, regions, rotation=15)
plt.ylim(0, max([metrics_full['mse'], metrics_noise['mse'], metrics_struct['mse']]) * 1.1)

# PSNR 비교
plt.subplot(3, 4, 10); plt.title("PSNR Comparison")
plt.bar(x, [metrics_full['psnr'], metrics_noise['psnr'], metrics_struct['psnr']], 
        width=width, color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('PSNR (dB)')
plt.xticks(x, regions, rotation=15)
plt.ylim(0, max([metrics_full['psnr'], metrics_noise['psnr'], metrics_struct['psnr']]) * 1.1)

# SSIM 비교
plt.subplot(3, 4, 11); plt.title("SSIM Comparison")
plt.bar(x, [metrics_full['ssim'], metrics_noise['ssim'], metrics_struct['ssim']], 
        width=width, color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('SSIM')
plt.xticks(x, regions, rotation=15)
plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()


# 11. Save result
save_path = "result/metrics.png"
pred_pil = Image.fromarray(sharpened_img)
pred_pil.save(save_path)
print(f"✅ Clean image saved: {save_path}")

# Save metrics to text file
with open("result/metrics_results.txt", "w") as f:
    f.write("DENOISING EVALUATION RESULTS\n")
    f.write("="*50 + "\n")
    f.write(f"Full Image - MSE: {metrics_full['mse']:.4f}, PSNR: {metrics_full['psnr']:.2f}, SSIM: {metrics_full['ssim']:.4f}\n")
    f.write(f"Noise ROI  - MSE: {metrics_noise['mse']:.4f}, PSNR: {metrics_noise['psnr']:.2f}, SSIM: {metrics_noise['ssim']:.4f}\n")
    f.write(f"Struct ROI - MSE: {metrics_struct['mse']:.4f}, PSNR: {metrics_struct['psnr']:.2f}, SSIM: {metrics_struct['ssim']:.4f}\n")
print("✅ Metrics saved to result/metrics_results.txt")
