import torch
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from noise.train.res_att_unet import AttResUNet
from noise.train.utils import load_model, generate_rois
import torch.nn.functional as F
from tqdm import tqdm


# 1. Config
model_path = "../model/denosing.pth"
test_folder_path = "../01-1.정식개방데이터/Test/01.원천데이터/VS_STR"
gt_folder_path = "../01-1.정식개방데이터/Test/00.클린데이터/VL_STR"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = (512, 512)


def preprocess_for_lpips(tensor):
    """[1,1,H,W] [0,1] → [1,3,H,W] [-1,1] 변환"""
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)  # Grayscale → RGB
    tensor = tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    return tensor


def compute_metrics_roi(preds, targets, roi_mask=None, data_range=1.0, device="cpu"):
    """ROI 기반 MSE/PSNR/SSIM/LPIPS 계산"""
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

        # MSE: ROI 안 픽셀의 평균
        valid = roi_mask.sum() * preds.shape[1] + 1e-8
        mse = ((preds_roi - targets_roi) ** 2).sum() / valid
    else:
        preds_roi = preds
        targets_roi = targets
        mse = F.mse_loss(preds_roi, targets_roi, reduction="mean")

    # Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    psnr = psnr_metric(preds_roi, targets_roi)
    ssim = ssim_metric(preds_roi, targets_roi)

    # LPIPS용 전처리 및 계산: 오류 발생 시 1.0 반환
    try:
        preds_lpips = preprocess_for_lpips(preds_roi)
        targets_lpips = preprocess_for_lpips(targets_roi)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
        lpips = lpips_metric(preds_lpips, targets_lpips)
    except Exception as e:
        print(f"⚠️ LPIPS 계산 오류: {e}")
        lpips = torch.tensor(1.0, device=device)

    return {
        "mse": mse.item(),
        "psnr": psnr.item(),
        "ssim": ssim.item(),
        "lpips": lpips.item(),
    }


# 2. Load model
print("🔄 Loading model...")
model = AttResUNet(in_channels=1, out_channels=1)
model = load_model(model, model_path, device)
model.eval()
print("✅ Model loaded successfully!")


# 3. 폴더에서 이미지 쌍 찾기 & GT 파일 매칭 가능한 것만 필터링
test_files = sorted([f for f in os.listdir(test_folder_path) if f.endswith(('.png', '.PNG', '.jpg', '.JPG'))])
valid_files = []

for test_file in test_files:
    gt_file = test_file.lower()
    gt_path = os.path.join(gt_folder_path, gt_file)
    if not os.path.exists(gt_path):
        alt_gt_file = test_file.replace('.PNG', '.png')
        gt_path = os.path.join(gt_folder_path, alt_gt_file)
    if os.path.exists(gt_path):
        valid_files.append(test_file)

print(f"📁 Found {len(test_files)} test images → {len(valid_files)} valid pairs")
if len(valid_files) == 0:
    print("❌ No valid pairs found! Check folder paths and file names.")
    exit()


# 4. 메트릭스 저장 리스트 초기화
full_mse, full_psnr, full_ssim, full_lpips = [], [], [], []
noise_mse, noise_psnr, noise_ssim, noise_lpips = [], [], [], []
struct_mse, struct_psnr, struct_ssim, struct_lpips = [], [], [], []

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])


# 5. 배치 평가 루프
print(f"🚀 Starting batch evaluation on {len(valid_files)} images...")
for i, test_file in enumerate(tqdm(valid_files, desc="🧹 Denoising", unit="img")):
    test_path = os.path.join(test_folder_path, test_file)
    gt_file = test_file.lower()
    gt_path = os.path.join(gt_folder_path, gt_file.replace('.PNG', '.png'))

    # Load images
    input_img = Image.open(test_path).convert("L")
    gt_img = Image.open(gt_path).convert("L")

    # 전처리
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    gt_img_resized = gt_img.resize(IMG_SIZE, resample=Image.BILINEAR)
    gt_img_np = np.array(gt_img_resized)
    orig_img_np = np.array(input_img.resize(IMG_SIZE, resample=Image.BILINEAR))

    # 추론
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        clean_image = output[0, 0].cpu().numpy()
        clean_image_uint8 = (clean_image * 255).astype(np.uint8)

    # 샤프닝
    sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img = cv2.filter2D(clean_image_uint8, -1, sharp_kernel)
    pred_resized_np = np.array(Image.fromarray(sharpened_img).resize(IMG_SIZE, resample=Image.BILINEAR))

    # 메트릭 계산용 텐서
    preds = torch.from_numpy(pred_resized_np / 255.0).float().unsqueeze(0).unsqueeze(0)
    targets = torch.from_numpy(gt_img_np / 255.0).float().unsqueeze(0).unsqueeze(0)

    # ROIs 생성 및 메트릭 계산
    try:
        noise_roi, struct_roi = generate_rois(orig_img_np, gt_img_np)

        metrics_full = compute_metrics_roi(preds, targets, roi_mask=None, device=device)
        metrics_noise = compute_metrics_roi(preds, targets, roi_mask=noise_roi, device=device)
        metrics_struct = compute_metrics_roi(preds, targets, roi_mask=struct_roi, device=device)

        # 누적
        full_mse.append(metrics_full['mse']); full_psnr.append(metrics_full['psnr'])
        full_ssim.append(metrics_full['ssim']); full_lpips.append(metrics_full['lpips'])
        noise_mse.append(metrics_noise['mse']); noise_psnr.append(metrics_noise['psnr'])
        noise_ssim.append(metrics_noise['ssim']); noise_lpips.append(metrics_noise['lpips'])
        struct_mse.append(metrics_struct['mse']); struct_psnr.append(metrics_struct['psnr'])
        struct_ssim.append(metrics_struct['ssim']); struct_lpips.append(metrics_struct['lpips'])

    except Exception as e:
        tqdm.write(f"❌ Error processing {test_file}: {e}")
        continue


# 6. 평균 및 표준편차 계산 및 출력
def print_avg_metrics(name, mse_list, psnr_list, ssim_list, lpips_list):
    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    print(f"📈 {name:15} MSE: {avg_mse:.4f} | PSNR: {avg_psnr:.2f}±{std_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
    return avg_mse, avg_psnr, std_psnr, avg_ssim, avg_lpips

n_images = len(full_mse)
print(f"\n🎉 Processed {n_images} images successfully!")

print("\n" + "=" * 70)
print("🎉 BATCH EVALUATION RESULTS (N=%d)" % n_images)
print("=" * 70)
full_avg = print_avg_metrics("Full Image", full_mse, full_psnr, full_ssim, full_lpips)
noise_avg = print_avg_metrics("Noise ROI", noise_mse, noise_psnr, noise_ssim, noise_lpips)
struct_avg = print_avg_metrics("Structure ROI", struct_mse, struct_psnr, struct_ssim, struct_lpips)
print("=" * 70)


# 7. 그래프 그리기 
regions = ['Full Image', 'Noise ROI', 'Structure ROI']
x = np.arange(len(regions))
width = 0.6

plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.title("MSE Comparison (Avg)")
plt.bar(x, [full_avg[0], noise_avg[0], struct_avg[0]], width=width,
        color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('MSE')
plt.xticks(x, regions, rotation=15)

plt.subplot(2, 3, 2)
plt.title("PSNR Comparison (Avg ± Std)")
plt.bar(x, [full_avg[1], noise_avg[1], struct_avg[1]], width=width,
        color=['skyblue', 'orange', 'lightgreen'], alpha=0.8,
        yerr=[full_avg[2], noise_avg[2], struct_avg[2]], capsize=5)
plt.ylabel('PSNR (dB)')
plt.xticks(x, regions, rotation=15)

plt.subplot(2, 3, 3)
plt.title("SSIM Comparison (Avg)")
plt.bar(x, [full_avg[3], noise_avg[3], struct_avg[3]], width=width,
        color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('SSIM')
plt.xticks(x, regions, rotation=15)

plt.subplot(2, 3, 4)
plt.title("LPIPS Comparison (Avg, Lower is Better)")
plt.bar(x, [full_avg[4], noise_avg[4], struct_avg[4]], width=width,
        color=['skyblue', 'orange', 'lightgreen'], alpha=0.8)
plt.ylabel('LPIPS ↓')
plt.xticks(x, regions, rotation=15)

plt.subplot(2, 3, 5)
plt.title("PSNR Distribution")
plt.boxplot([full_psnr, noise_psnr, struct_psnr], labels=regions)
plt.ylabel('PSNR (dB)')
plt.xticks(rotation=15)

plt.subplot(2, 3, 6)
plt.title("LPIPS Distribution (Lower is Better)")
plt.boxplot([full_lpips, noise_lpips, struct_lpips], labels=regions)
plt.ylabel('LPIPS ↓')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("./result/batch_metrics.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. 결과 파일 저장
os.makedirs("result", exist_ok=True)
with open("./result/batch_metrics.txt", "w") as f:
    f.write(f"BATCH EVALUATION RESULTS (N={n_images}) - LPIPS 포함\n")
    f.write("="*70 + "\n")
    f.write(f"Full Image     - MSE: {full_avg[0]:.4f}, PSNR: {full_avg[1]:.2f}±{full_avg[2]:.2f}, SSIM: {full_avg[3]:.4f}, LPIPS: {full_avg[4]:.4f}\n")
    f.write(f"Noise ROI      - MSE: {noise_avg[0]:.4f}, PSNR: {noise_avg[1]:.2f}±{noise_avg[2]:.2f}, SSIM: {noise_avg[3]:.4f}, LPIPS: {noise_avg[4]:.4f}\n")
    f.write(f"Structure ROI  - MSE: {struct_avg[0]:.4f}, PSNR: {struct_avg[1]:.2f}±{struct_avg[2]:.2f}, SSIM: {struct_avg[3]:.4f}, LPIPS: {struct_avg[4]:.4f}\n")

print("✅ Batch results saved to result/batch_metrics.txt & result/batch_metrics.png")
