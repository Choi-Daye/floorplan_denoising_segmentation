import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def draw_line_on_image(image_path, json_path, target_id):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # json 파일 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found at {json_path}")
        return

    # target_id에 해당하는 annotation 찾기
    annotations_matching = [ann for ann in labels['annotations'] if ann['category_id'] == target_id]
    
    # 해당하는 target_id가 존재하지 않을 때
    if not annotations_matching:
        print(f"No category_id : {target_id}")
        return

    # 해당하는 target_id가 존재할 때
    for annotation in annotations_matching:
        # segmentation이 존재할 때
        if annotation['segmentation']:
            seg = annotation['segmentation']

            for poly in seg:
                pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
        
        # bounding box만 존재할 때
        elif 'bbox' in annotation and annotation['bbox']:
            x, y, w, h = annotation['bbox']
            # 네 꼭짓점 좌표 계산
            pts = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]).astype(np.int32)
            # 폴리라인(사각형) 그리기
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

        else:
            print("No information")

    # 출력
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()





# 실행
image_path = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR/APT_FP_STR_952297789.PNG"
json_path = "../01-1.정식개방데이터/Validation/02.라벨링데이터/VL_STR/APT_FP_STR_952297789.json"

draw_line_on_image(image_path, json_path, 12)    # 9 : 구조_출입문, 10 : 구조_창호, 11 : 구조_벽체, 12 : background