import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from noise.train.res_att_unet import AttResUNet
from utils import load_model
import cv2
import numpy as np
from tqdm import tqdm


input_folder = "../01-1.정식개방데이터/Test/01.원천데이터/VS_STR"
output_folder = "../01-1.정식개방데이터/Test/03.클린데이터/VS_STR"
model_path = "../model/denoising.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 모델 로드
model = AttResUNet(in_channels=1, out_channels=1)
model = load_model(model, model_path, device)
model.eval()

# 이미지 파일 리스트
image_files = glob(os.path.join(input_folder, "*.PNG"))

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

os.makedirs(output_folder, exist_ok=True)

# 샤프닝 필터 커널
sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

for img_path in tqdm(image_files, desc="Processing images"):
    img = Image.open(img_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        clean_image = output[0,0].cpu().numpy()
        clean_image_uint8 = (clean_image * 255).astype('uint8')

        # 샤프닝 필터 적용
        sharpened_img = cv2.filter2D(clean_image_uint8, -1, sharp_kernel)

        cleaned_pil = Image.fromarray(sharpened_img)
        # 원본 크기로 복원
        cleaned_pil = cleaned_pil.resize(img.size, resample=Image.BILINEAR)
        
        base_name = os.path.basename(img_path).split('.')[0]
        cleaned_path = os.path.join(output_folder, base_name + ".png")
        cleaned_pil.save(cleaned_path)

print("finished")
