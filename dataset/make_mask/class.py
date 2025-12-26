import cv2
import numpy as np
import json
import os
from tqdm import tqdm


def make_mask(json_path):
    # json 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found at {json_path}")
        return None

    # image size
    width, height = labels['images'][0]['width'], labels['images'][0]['height']

    # 빈 mask 생성
    mask = np.full((height, width), 0, dtype=np.uint8)

    for annotation in labels['annotations']:
        category_id = annotation['category_id']

        # category_id에 따른 마스크 값 지정
        if category_id == 9:        # 구조_출입문
            mask_val = 1
        elif category_id == 10:     # 구조_창호
            mask_val = 2
        elif category_id == 11:     # 구조_벽체
            mask_val = 3
        elif category_id == 12:  # background -> 제외
            continue               
        else:
            continue

        # segmentation
        if annotation['segmentation']:
            seg = annotation['segmentation']
            for poly in seg:
                pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], mask_val)
        # bounding box
        elif annotation['bbox']:
            x, y, w, h = map(int, annotation['bbox'])
            cv2.rectangle(mask, (x, y), (x + w, y + h), mask_val, thickness=-1)
        else:
            print("No segmentation or bbox in annotation")

    return mask


def save_mask(json_path):
    # 마스크 생성
    mask = make_mask(json_path)
    if mask is None:
        return None

    # 폴더명 추출
    dir = os.path.dirname(json_path)
    # 파일명 추출
    file_name = os.path.splitext(os.path.basename(json_path))[0]     # basenmae : 파일명만 반환
    # 폴더명 변환
    new_path = dir.replace("02.라벨링데이터", "03.마스크데이터/구조_출입문") 
    os.makedirs(new_path, exist_ok=True)    # 폴더 생성
    # 전체 경로 만들기
    save_path = os.path.join(new_path, f"{file_name}.png")

    # 저장 
    cv2.imwrite(save_path, mask)

    return (f"Saved mask: {save_path}")


def batch_make_masks(root_dir, start_file=None):    # root_dir : 루트 폴더
    json_list = []
    # 모든 하위 폴더 및 파일 순회
    for dirpath, dirnames, filenames in os.walk(root_dir):      # 폴더 경로, 하위 폴더 리스트, 파일 리스트
        for filename in filenames:
            if filename.endswith('.json'):      # json 파일만
                json_list.append(os.path.join(dirpath, filename))   # json_list에 전체 경로 추가
    json_list.sort()  # 동일한 순서로 처리

    start_idx = 0
    # 중간부터 시작
    if start_file:
        for idx, path in enumerate(json_list):
            if os.path.basename(path) == start_file:    # 파일명 일치
                start_idx = idx     # idx 가져오기
                break

    for json_path in tqdm(json_list[start_idx:], desc="Processing JSON files"):
        save_mask(json_path)




# 실행
root_dir = "../01-1.정식개방데이터/Training/02.라벨링데이터/TL_STR"
batch_make_masks(root_dir)