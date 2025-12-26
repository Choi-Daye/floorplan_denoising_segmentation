import argparse
import torch
import os
import csv
from datetime import datetime
from noise.train.res_att_unet import AttResUNet
from dataset import get_loaders
from noise.train.train import train_one_epoch, validate, WeightedComboSegLoss
from noise.train.utils import set_seed, save_model, plot_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import CONFIG


def main(args):
    best_val_loss = float('inf')
    best_metrics = {}

    # Set seed
    set_seed(args.seed)
    print(f"🌱 Random seed: {args.seed}")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 Device: {device}")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    

    # Training data loader
    print("=" * 60)
    print("📚 TRAINING DATASET")
    train_loader = get_loaders(
        args.train_image_dir, 
        args.train_clean_dir,  
        args.train_mask_dir,
        args.batch_size,
        shuffle=True
    )
    
    # Validation data loader (REQUIRED)
    print("=" * 60)
    print("📊 VALIDATION DATASET")
    if not args.val_image_dir or not args.val_clean_dir:  
        raise ValueError("Validation dataset paths are required! Please provide --val_image_dir and --val_clean_dir")
    
    val_loader = get_loaders(
        args.val_image_dir,
        args.val_clean_dir,
        args.val_mask_dir, 
        args.batch_size,
        shuffle=False
    )
    

    # Model creation
    print(f"\n🏗️  Creating model...")
    model = AttResUNet(in_channels=1, out_channels=1).to(device).float()
    
    # Loss & Optimizer
    class_weights = torch.tensor([0.1, 0.3, 1.0, 1.0], dtype=torch.float32).to(device)      # class id 별로 가중치 부여
    criterion = WeightedComboSegLoss(class_weights=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stop_patience = 10        # 10번 이상 성능 개선이 안 될 경우, 중단
    early_stop_counter = 0
    best_val_loss = float('inf')
    

    print(f"\n⚙️  Training configuration:")
    print(f"   - Optimizer: {optimizer}")
    print(f"   - Learning Rate: {args.lr}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Criterion: {args.criterion}")
    print(f"   - Train samples: {len(train_loader.dataset)}")
    print(f"   - Validation samples: {len(val_loader.dataset)}")


    # 폴더 존재 확인
    save_dir = os.path.dirname(args.save_path)
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs("result", exist_ok=True)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_ssim': [],   
        'val_psnr': [],  
        'lr': [],
    }
    
    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"./result/training_log_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Train_Loss', 'Val_Loss',
            'Val_SSIM', 'Val_PSNR', 
            'lr'
        ])
    
    print(f"\n{'='*60}")
    print(f"🎯 Training started!")
    

    # Training loop
    for epoch in range(args.epochs):
        # Train on TRAINING dataset 
        train_loss = train_one_epoch(
            model, train_loader, class_weights, criterion, optimizer, device, epoch, args.epochs
        )

        # Validate on VALIDATION dataset (all metrics)
        result = validate(model, val_loader, criterion, class_weights, device)
        val_loss, val_ssim, val_psnr = result

        # early stopping
        scheduler.step(val_loss)
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")

        if val_loss < best_val_loss:        # 최적의 loss 저장
            best_val_loss = val_loss
            early_stop_counter = 0
            save_model(model, args.save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"🔔 Early stopping at epoch {epoch+1}")
            break


        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ssim'].append(val_ssim)
        history['val_psnr'].append(val_psnr)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"📊 Epoch {epoch+1}/{args.epochs} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val   Loss: {val_loss:.4f}")
        print(f"   Val   SSIM: {val_ssim:.4f}")
        print(f"   Val   PSNR: {val_psnr:.4f}")
        
        # Save best model based on Dice Score
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'loss': val_loss,
                'ssim': val_ssim,
                'dice': val_psnr,
                'epoch': epoch + 1
            }
            save_model(model, args.save_path)
            print(f"   ✨ Best model saved! (Val Loss: {best_val_loss:.4f})")
        
        print(f"{'='*60}\n")
    
        # Save CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_ssim:.6f}", f"{val_psnr:.6f}", 
                f"{optimizer.param_groups[0]['lr']:.6e}"
            ])
    

    # Save plot
    plot_metrics(history, save_path=f"./result/training_metrics_{timestamp}.png")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"🎉 Training completed!")
    print(f"{'='*60}")
    print(f"📁 Best model: {args.save_path}")
    print(f"\n🏆 Best Perf Metrics at epoch {best_metrics['epoch']}:")
    print(f"- Combo Score: {best_metrics['score']:.4f}")
    print(f"- Struct Dice: {best_metrics['dice_struct']:.4f}, Struct IoU: {best_metrics['iou_struct']:.4f}")
    print(f"- Noise PSNR: {best_metrics['psnr_noise']:.2f} dB")
    print(f"- Struct SSIM: {best_metrics['ssim_struct']:.4f}")
    print(f"- Val Loss:   {best_metrics['loss']:.4f}")
    print(f"\n📂 Output Files:\n- CSV: {csv_filename}\n- Plot: training_metrics_{timestamp}.png")
    print(f"{'='*60}\n")




if __name__ == "__main__":
    args = argparse.Namespace(**CONFIG)
    main(args)
