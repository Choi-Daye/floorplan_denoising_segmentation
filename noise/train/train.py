import torch
import torch.nn as nn
from tqdm import tqdm
import kornia
from noise.train.utils import psnr_score, ssim_score, generate_rois


class WeightedComboSegLoss(nn.Module):
    def __init__(self, class_weights, ssim_weight=0.5, dice_weight=1.0, boundary_weight=0.2):
        super().__init__()
        self.class_weights = class_weights          # torch.Tensor, shape [num_classes]
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        self.ssim_weight = ssim_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def dice_loss(self, inputs, targets, mask_weights, smooth=1e-7):
        inputs = torch.sigmoid(inputs)      # logits을 확률로 변환
        inputs_flat = inputs.view(-1)       # 모든 픽셀 펼침
        targets_flat = targets.view(-1)
        mask_weights_flat = mask_weights.view(-1)
        
        inputs_weighted = inputs_flat * mask_weights_flat
        targets_weighted = targets_flat * mask_weights_flat

        intersection = (inputs_weighted * targets_weighted).sum()
        denominator = inputs_weighted.sum() + targets_weighted.sum()

        dice = (2. * intersection + smooth) / (denominator + smooth)

        return 1 - dice

    # Pixel-wise weighted BCE
    def weighted_bce_loss(self, inputs, targets, mask_weights):
        bce_pixel = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )  # shape [B, 1, H, W]
        bce_pixel = bce_pixel.squeeze(1)
        loss = (bce_pixel * mask_weights).mean()

        return loss

    def forward(self, inputs, targets, masks):
        # inputs: logits [B, 1, H, W], targets: [B, 1, H, W], masks: [B, H, W]
        mask_weights = self.class_weights[masks]  # shape [B, H, W]

        bce = self.weighted_bce_loss(inputs, targets, mask_weights)
        dice = self.dice_loss(inputs, targets, mask_weights)
        ssim = self.ssim_loss(torch.sigmoid(inputs), targets)
        boundary = 0.0

        return bce + self.dice_weight * dice + self.ssim_weight * ssim + self.boundary_weight * boundary


def train_one_epoch(model, loader, class_weights, criterion, optimizer, device, epoch, total_epochs):
    """
    Train for one epoch on TRAINING dataset
    Returns: average training loss
    """
    model.train()
    running_loss = 0.0
    
    # tqdm progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [TRAIN]", 
                ncols=100, leave=True)
    # class_weights = torch.tensor([0.3, 1.0, 3.0, 3.0], dtype=torch.float32).to(device)

    for batch_idx, (inputs, cleans, masks) in enumerate(pbar):
        inputs, cleans, masks = inputs.to(device).float(), cleans.to(device).float(), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, cleans, masks)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(loader)
    
    return avg_loss


def validate(model, loader, criterion, class_weights, device):
    """
    Validate on VALIDATION dataset
    Calculate all metrics for both Noise ROI and Structure ROI:
    Loss, Dice, IoU, PSNR, SSIM
    """
    model.eval()
    running_loss = 0.0
    ssim_values = []
    psnr_values = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="              [VAL]  ", ncols=120, leave=False)
        for inputs, cleans, masks in pbar:
            inputs, cleans, masks = inputs.to(device), cleans.to(device), masks.to(device)
        
            outputs = model(inputs)

            loss = criterion(outputs, cleans, masks)
            running_loss += loss.item()

            batch_size = inputs.shape[0]
            for i in range(batch_size):
                orig_img_np = inputs[i].cpu().numpy().squeeze() * 255.0
                gt_img_np = cleans[i].cpu().numpy().squeeze() * 255.0
                pred = torch.sigmoid(outputs[i]).cpu().numpy().squeeze()
                
                noise_roi, struct_roi = generate_rois(orig_img_np, gt_img_np)
                
                # PSNR, SSIM 계산 - 구조 ROI로만 예시
                psnr_val = psnr_score(torch.tensor(pred), torch.tensor(gt_img_np / 255.), struct_roi)
                ssim_val = ssim_score(torch.tensor(pred), torch.tensor(gt_img_np / 255.), struct_roi)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                
        avg_loss = running_loss / len(loader)
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        
    return avg_loss, avg_ssim, avg_psnr