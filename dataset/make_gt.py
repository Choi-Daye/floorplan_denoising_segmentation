import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def draw_outside_mask(image_path, json_path, target_ids):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # JSON 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found at {json_path}")
        return None

    # 관심 category_id 있는 라벨 필터링
    annotations_matching = [ann for ann in labels['annotations'] if ann['category_id'] in target_ids]
    if not annotations_matching:
        print(f"No category_id in {target_ids} for {json_path}")
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # segmentation 또는 bbox 활용해 마스크 채우기
    for annotation in annotations_matching:
        if annotation.get('segmentation'):
            seg = annotation['segmentation']
            for poly in seg:
                pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], color=255)
        elif 'bbox' in annotation and annotation['bbox']:
            x, y, w, h = annotation['bbox']
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=255)

    # 마스크 바깥 흰색으로 처리
    masked_image = image.copy()
    masked_image[mask == 0] = [255, 255, 255]

    return masked_image

def save_mask(image_path, json_path, target_ids):
    masked_image = draw_outside_mask(image_path, json_path, target_ids)
    if masked_image is None:
        return False

    dir = os.path.dirname(json_path)
    file_name = os.path.splitext(os.path.basename(json_path))[0]

    # 저장 폴더: "02.라벨링데이터" → "00.클린데이터" 변경
    save_folder = dir.replace("02.라벨링데이터", "00.클린데이터")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{file_name}.png")

    # RGB → BGR 변환 후 저장
    save_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img)
    return True

def batch_draw_outside_mask(json_root_dir, image_root_dir, target_ids, start_file=None):
    json_list = []
    for dirpath, dirnames, filenames in os.walk(json_root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                json_list.append(os.path.join(dirpath, filename))
    json_list.sort()

    start_idx = 0
    if start_file:
        for idx, path in enumerate(json_list):
            if os.path.basename(path) == start_file:
                start_idx = idx
                break

    for json_path in tqdm(json_list[start_idx:], desc="Processing JSON files"):
        file_name = os.path.splitext(os.path.basename(json_path))[0]

        # 이미지 경로 추정(PNG, 대문자 확장자)
        image_path = os.path.join(image_root_dir, file_name + '.PNG')
        if not os.path.exists(image_path):
            # 확장자 다를 경우 소문자 png 체크
            image_path = os.path.join(image_root_dir, file_name + '.png')
            if not os.path.exists(image_path):
                print(f"Image file not found for JSON: {json_path}")
                continue
        save_mask(image_path, json_path, target_ids)




# 실행
image_root = '../01-1.정식개방데이터/Test/01.원천데이터/VS_STR'
json_root = '../01-1.정식개방데이터/Test/02.라벨링데이터/VL_STR'
target_ids = [9, 10, 11]

batch_draw_outside_mask(json_root, image_root, target_ids)