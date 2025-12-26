import os
import glob
import json
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm 


# 0. IoU / 매칭 / AP 계산 함수들
def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def match_masks(gt_masks, pred_masks, iou_thr=0.5):
    gt_used = np.zeros(len(gt_masks), dtype=bool)
    y_true, y_pred, ious = [], [], []

    for pm in pred_masks:
        best_iou = 0.0
        best_gt = -1
        for i, gm in enumerate(gt_masks):
            if gt_used[i]:
                continue
            v = mask_iou(pm, gm)
            if v > best_iou:
                best_iou = v
                best_gt = i

        if best_iou >= iou_thr and best_gt >= 0:
            gt_used[best_gt] = True
            y_true.append(1)
            y_pred.append(1)
            ious.append(best_iou)
        else:
            y_true.append(0)
            y_pred.append(1)

    fn = (~gt_used).sum()
    y_true.extend([1] * fn)
    y_pred.extend([0] * fn)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1, ious


def ap_from_scores(gt_masks, pred_masks, pred_scores, iou_thr=0.5):
    if len(pred_masks) == 0:
        return 0.0

    order = np.argsort(-pred_scores)
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]

    gt_used = np.zeros(len(gt_masks), dtype=bool)
    tps, fps = [], []

    for pm in pred_masks:
        best_iou = 0.0
        best_gt = -1
        for i, gm in enumerate(gt_masks):
            if gt_used[i]:
                continue
            v = mask_iou(pm, gm)
            if v > best_iou:
                best_iou = v
                best_gt = i

        if best_iou >= iou_thr and best_gt >= 0:
            gt_used[best_gt] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps = np.array(tps)
    fps = np.array(fps)
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    recalls = cum_tp / max(len(gt_masks), 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return float(ap)


# 1. device & 모델 준비 
device = torch.device('cpu')
print("Device:", device)

num_classes = 4  # background + {1:Door, 2:Window, 3:Wall}

model = maskrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes
)

checkpoint = torch.load('detection/checkpoint/last_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()



# 2. 한 이미지에 대해 평가하는 함수
def evaluate_one(image_path, json_path, iou_thr=0.5, dilate=True):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    with open(json_path) as f:
        annotation = json.load(f)

    # GT masks
    gt_label_masks = []
    for ann in annotation["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in [9, 10, 11]:
            continue

        if cat_id == 9:
            l = 1
        elif cat_id == 10:
            l = 2
        elif cat_id == 11:
            l = 3

        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        seg = ann.get("segmentation", [])
        for poly in seg:
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, outline=1, fill=1)

        mask_np = np.array(mask_img).astype(bool)
        if mask_np.sum() == 0:
            continue

        if dilate:
            kernel = np.ones((5, 5), dtype=bool)
            mask_np = ndimage.binary_dilation(mask_np, structure=kernel, iterations=3)

        gt_label_masks.append((mask_np, l))

    gt_masks = [m for (m, l) in gt_label_masks]
    gt_labels = [l for (m, l) in gt_label_masks]

    # 예측
    image_tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        outputs = model([image_tensor])
    output = outputs[0]

    score_thr = 0.5
    scores = output['scores']
    keep = scores >= score_thr

    pred_masks = output['masks'][keep].cpu().numpy()  # [N,1,H,W]
    pred_masks = pred_masks[:, 0] > 0.5               # [N,H,W]
    pred_labels = output['labels'][keep].cpu().numpy()
    pred_scores = scores[keep].cpu().numpy()

    if len(gt_masks) == 0:
        return {
            "prec": 0.0, "rec": 0.0, "f1": 0.0,
            "ap50_all": 0.0,
            "ap50_door": 0.0, "ap50_window": 0.0, "ap50_wall": 0.0,
            "map50": 0.0
        }

    prec, rec, f1, ious = match_masks(gt_masks, pred_masks, iou_thr=iou_thr)

    if len(pred_masks) > 0:
        ap50_all = ap_from_scores(gt_masks, pred_masks, pred_scores, iou_thr=iou_thr)
    else:
        ap50_all = 0.0

    def class_ap50(class_id):
        gt_c = [m for m, l in zip(gt_masks, gt_labels) if l == class_id]
        pred_c = [m for m, l in zip(pred_masks, pred_labels) if l == class_id]
        scores_c = [s for s, l in zip(pred_scores, pred_labels) if l == class_id]
        if len(gt_c) == 0 or len(pred_c) == 0:
            return 0.0
        pred_c = np.stack(pred_c, axis=0)
        scores_c = np.array(scores_c)
        return ap_from_scores(gt_c, pred_c, scores_c, iou_thr=iou_thr)

    ap50_door   = class_ap50(1)
    ap50_window = class_ap50(2)
    ap50_wall   = class_ap50(3)
    map50 = (ap50_door + ap50_window + ap50_wall) / 3.0

    return {
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "ap50_all": float(ap50_all),
        "ap50_door": float(ap50_door),
        "ap50_window": float(ap50_window),
        "ap50_wall": float(ap50_wall),
        "map50": float(map50)
    }


# 3. 폴더 전체를 순회하면서 평가
img_dir  = '../01-1.정식개방데이터/Test/03.클린데이터/VS_STR'
json_dir = '../01-1.정식개방데이터/Test/02.라벨링데이터/VL_STR'

image_paths = sorted(
    glob.glob(os.path.join(img_dir, '*.png'))
) + sorted(
    glob.glob(os.path.join(img_dir, '*.PNG'))
)

metrics_list = []
per_image_results = {}

for img_path in tqdm(image_paths, desc="Evaluating images"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(json_dir, base + '.json')
    if not os.path.exists(json_path):
        print(f"[WARN] JSON not found for {img_path}, skip.")
        continue

    m = evaluate_one(img_path, json_path, iou_thr=0.5, dilate=True)
    metrics_list.append(m)
    per_image_results[base] = m  # 파일명 기준으로 저장


# 4. 전체 평균 계산
def mean_of(key):
    vals = [m[key] for m in metrics_list]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0

global_metrics = {
    "Precision@0.5IoU": mean_of('prec'),
    "Recall@0.5IoU": mean_of('rec'),
    "F1@0.5IoU": mean_of('f1'),
    "AP50_all": mean_of('ap50_all'),
    "AP50_Door": mean_of('ap50_door'),
    "AP50_Window": mean_of('ap50_window'),
    "AP50_Wall": mean_of('ap50_wall'),
    "mAP50": mean_of('map50')
}

print("=== Dataset metrics (mean over images) ===")
for k, v in global_metrics.items():
    print(f"{k}: {v:.4f}")


# 5. 결과 저장
os.makedirs('./result', exist_ok=True)
save_dict = {
    "global": global_metrics,
    "per_image": per_image_results
}
with open('./result/gt_clean.json', 'w', encoding='utf-8') as f:
    json.dump(save_dict, f, indent=2, ensure_ascii=False)
