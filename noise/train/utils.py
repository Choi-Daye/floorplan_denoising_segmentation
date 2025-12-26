import torch
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def save_model(model, path):
    """Save model"""
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load model"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded: {path}")
    return model


def generate_rois(orig_img_np, gt_img_np, thresh=80, dilate_iter=1):
    """
    노이즈 ROI = 원본 이미지의 선 - 정답 이미지의 선 
    구조 ROI = 정답 이미지의 선 영역
    """
    # 원본 ROI
    if orig_img_np.dtype != np.uint8:
        orig_img_uint8 = orig_img_np.astype(np.uint8)
    else:
        orig_img_uint8 = orig_img_np
    orig_edges = cv2.Canny(orig_img_uint8, thresh, thresh * 2)
    orig_edges = cv2.dilate(orig_edges, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    orig_mask = (orig_edges > 0).astype(np.uint8)

    # 정답 ROI
    if gt_img_np.dtype != np.uint8:
        gt_img_uint8 = gt_img_np.astype(np.uint8)
    else:
        gt_img_uint8 = gt_img_np
    gt_edges = cv2.Canny(gt_img_uint8, thresh, thresh * 2)
    gt_edges = cv2.dilate(gt_edges, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    gt_mask = (gt_edges > 0).astype(np.uint8)

    noise_roi = np.clip(orig_mask - gt_mask, 0, 1).astype(np.float32)
    struct_roi = gt_mask.astype(np.float32)

    return noise_roi, struct_roi


def dice_score(preds, targets, roi_mask, threshold=0.5, smooth=1e-7):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:
        roi_mask = roi_mask.unsqueeze(0)  # [1,H,W]
    if roi_mask.shape != preds.shape:
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    preds = preds * roi_mask
    targets = targets * roi_mask
    intersection = (preds * targets).sum(dim=(1,2,3))   # 교집합    # dim=(1,2,3): 한 배치 이미지 내 모든 픽셀 전체를 합산
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))   # 합집합
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_score(preds, targets, roi_mask, threshold=0.5, smooth=1e-7):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:     # [H,W]
        roi_mask = roi_mask.unsqueeze(0)        # [1,H,W]
    if roi_mask.shape != preds.shape:       # roi, preds 크기가 일치하지 않으면
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    preds = preds * roi_mask        # ROI mask가 1인 영역만
    targets = targets * roi_mask
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def precision_score(preds, targets, roi_mask, threshold=0.5, smooth=1e-7):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:
        roi_mask = roi_mask.unsqueeze(0)
    if roi_mask.shape != preds.shape:
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    preds = preds * roi_mask
    targets = targets * roi_mask

    true_positive = (preds * targets).sum(dim=(1, 2, 3))
    predicted_positive = preds.sum(dim=(1, 2, 3))
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.mean().item()


def recall_score(preds, targets, roi_mask, threshold=0.5, smooth=1e-7):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:
        roi_mask = roi_mask.unsqueeze(0)
    if roi_mask.shape != preds.shape:
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    preds = preds * roi_mask
    targets = targets * roi_mask

    true_positive = (preds * targets).sum(dim=(1, 2, 3))
    actual_positive = targets.sum(dim=(1, 2, 3))
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.mean().item()


def psnr_score(preds, targets, roi_mask, threshold=0.5, data_range=1.0):
    # preds, targets는 0~1 float 값으로 가정
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:
        roi_mask = roi_mask.unsqueeze(0)
    if roi_mask.shape != preds.shape:
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    
    preds_masked = preds * roi_mask
    targets_masked = targets * roi_mask
    
    preds_np = preds_masked.cpu().numpy()
    targets_np = targets_masked.cpu().numpy()
    
    psnr_values = []
    for pred, target in zip(preds_np, targets_np):
        pred = pred.squeeze()
        target = target.squeeze()
        mask = roi_mask.cpu().numpy().squeeze()
        if mask.sum() > 0:
            try:
                psnr = peak_signal_noise_ratio(target / 255.0, pred / 255.0, data_range=1.0)
                psnr_values.append(psnr)
            except:
                psnr_values.append(0.0)
    return np.mean(psnr_values) if psnr_values else 0.0


def ssim_score(preds, targets, roi_mask, threshold=0.5, data_range=1.0):
    roi_mask = torch.from_numpy(roi_mask).to(preds.device)
    if roi_mask.dim() == 2:
        roi_mask = roi_mask.unsqueeze(0)
    if roi_mask.shape != preds.shape:
        roi_mask = torch.nn.functional.interpolate(roi_mask.unsqueeze(0), size=preds.shape[-2:], mode='nearest').squeeze(0)
    
    preds_masked = preds * roi_mask
    targets_masked = targets * roi_mask
    
    preds_np = preds_masked.cpu().numpy()
    targets_np = targets_masked.cpu().numpy()
    
    ssim_values = []
    for pred, target in zip(preds_np, targets_np):
        pred = pred.squeeze()
        target = target.squeeze()
        mask = roi_mask.cpu().numpy().squeeze()
        if mask.sum() > 0:
            try:
                ssim = structural_similarity(target, pred, data_range=data_range)
                ssim_values.append(ssim)
            except:
                ssim_values.append(1.0)
    return np.mean(ssim_values) if ssim_values else 0.0


def plot_metrics(history, save_path='result/training_metrics.png'):
    """구조 ROI와 노이즈 ROI에 대해 주요 지표를 동시 시각화"""
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    epochs = history['epoch']

    metric_keys = ['dice', 'iou', 'psnr', 'ssim']
    roi_types = ['struct', 'noise']
    titles = {
        'dice': 'Dice',
        'iou': 'IoU',
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'precision': 'Precision',
        'recall': 'Recall',
    }

    for j, roi in enumerate(roi_types):
        for i, key in enumerate(metric_keys):
            metric_hist = history[f'val_{key}_{roi}']
            axes[j, i].plot(epochs, metric_hist, linewidth=2)
            axes[j, i].set_title(f"{titles[key]} ({roi.capitalize()} ROI)", fontsize=14, fontweight='bold')
            axes[j, i].set_xlabel('Epoch', fontsize=12)
            axes[j, i].set_ylabel(titles[key], fontsize=12)
            axes[j, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Training plot saved: {save_path}")
