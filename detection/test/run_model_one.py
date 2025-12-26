import torch
import torchvision
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_score, recall_score, f1_score


# 0. IoU / 매칭 / AP 계산 함수들
def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0


def match_masks(gt_masks, pred_masks, iou_thr=0.5):
    gt_used = np.zeros(len(gt_masks), dtype=bool)
    y_true = []
    y_pred = []
    ious = []

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
    tps = []
    fps = []

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


# 1. device = CPU 고정
device = torch.device('cpu')
print("Device:", device)


# 2. 모델 준비
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

checkpoint = torch.load('detection/checkpoint/origin.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


# 3. 이미지 & JSON 로드
image_path = '../01-1.정식개방데이터/Test/01.원천데이터/VS_STR/APT_FP_STR_000608681.PNG'
json_path  = '../01-1.정식개방데이터/Test/02.라벨링데이터/VL_STR/APT_FP_STR_000608681.json'

image = Image.open(image_path).convert("RGB")
w, h = image.size

with open(json_path) as f:
    annotation = json.load(f)


# 4. GT mask 만들기 (+ 강한 dilation)
gt_label_masks = []  # (mask_bool, label_id)

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

    # GT 경계 두껍게: 5x5 커널, 3회 팽창
    kernel = np.ones((5, 5), dtype=bool)
    dilated = ndimage.binary_dilation(mask_np, structure=kernel, iterations=3)
    mask_np = dilated

    gt_label_masks.append((mask_np, l))

gt_masks = [m for (m, l) in gt_label_masks]
gt_labels = [l for (m, l) in gt_label_masks]
print("Num GT instances:", len(gt_masks))


# 5. 모델 예측
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
print("Num Pred instances:", len(pred_masks))


# 6. 지표 계산 (Precision / Recall / F1 / AP50 / mAP50)
# 6-1) 전체 기준 Precision / Recall / F1 (IoU >= 0.5)
prec, rec, f1, ious = match_masks(gt_masks, pred_masks, iou_thr=0.5)
print(f"Precision@0.5IoU: {prec:.4f}")
print(f"Recall@0.5IoU   : {rec:.4f}")
print(f"F1-score@0.5IoU : {f1:.4f}")
print(f"Mean IoU (TP만) : {np.mean(ious) if len(ious) > 0 else 0:.4f}")

# 6-2) 전체 AP50
if len(gt_masks) > 0 and len(pred_masks) > 0:
    ap50_all = ap_from_scores(gt_masks, pred_masks, pred_scores, iou_thr=0.5)
else:
    ap50_all = 0.0
print(f"AP50 (all classes): {ap50_all:.4f}")

# 6-3) 클래스별 AP50, mAP50
def class_ap50(class_id):
    gt_c = [m for m, l in zip(gt_masks, gt_labels) if l == class_id]
    pred_c = [m for m, l in zip(pred_masks, pred_labels) if l == class_id]
    scores_c = [s for s, l in zip(pred_scores, pred_labels) if l == class_id]
    if len(gt_c) == 0 or len(pred_c) == 0:
        return 0.0
    gt_c = list(gt_c)
    pred_c = np.stack(pred_c, axis=0)
    scores_c = np.array(scores_c)
    return ap_from_scores(gt_c, pred_c, scores_c, iou_thr=0.5)

ap50_door   = class_ap50(1)
ap50_window = class_ap50(2)
ap50_wall   = class_ap50(3)
map50 = (ap50_door + ap50_window + ap50_wall) / 3.0

print(f"AP50 Door   : {ap50_door:.4f}")
print(f"AP50 Window : {ap50_window:.4f}")
print(f"AP50 Wall   : {ap50_wall:.4f}")
print(f"mAP50       : {map50:.4f}")


# 7. 컬러 마스크 + 시각화
def build_color_mask(boolean_masks):
    if len(boolean_masks) == 0:
        return np.zeros((h, w, 3), dtype=np.float32)
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.0, 1.0, size=(len(boolean_masks), 3))
    for i, m in enumerate(boolean_masks):
        color_mask[m] = colors[i]
    return color_mask

gt_color_mask = build_color_mask(gt_masks)
pred_color_mask = build_color_mask(pred_masks)

img_np = np.array(image)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes[0].imshow(img_np)
axes[0].imshow(gt_color_mask, alpha=0.5)
axes[0].set_title("GT masks (dilated)")
axes[0].axis('off')

axes[1].imshow(img_np)
axes[1].imshow(pred_color_mask, alpha=0.5)
axes[1].set_title("Pred masks")
axes[1].axis('off')

plt.tight_layout()
plt.show()