import os
import json
from PIL import Image
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 配置
IMG_DIR = '/home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/test/images'
LABEL_DIR = '/home/qi.xiong/Dataset/UAV_Sheep/yolo_dataset/test/labels'
CLASS_NAMES = ['sheep']
MODEL_PATH = '/home/qi.xiong/Improve/yolov13/yolov13_UAV_Sheep/tokenc3_lsnet_p2/weights/best.pt'
COCO_ANN_PATH = 'annotations.json'
COCO_RES_PATH = 'results.json'

# 1. YOLO标签转COCO格式
def yolo_to_coco(yolo_dir, img_dir, class_names, output_json):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(yolo_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue
        img = Image.open(img_path)
        width, height = img.size
        images.append({
            "file_name": img_name,
            "height": height,
            "width": width,
            "id": img_id
        })
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                x1 = (x - w / 2) * width
                y1 = (y - h / 2) * height
                w_box = w * width
                h_box = h * height
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x1, y1, w_box, h_box],
                    "area": w_box * h_box,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1
    categories = [{"id": i+1, "name": name} for i, name in enumerate(class_names)]
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=4)
    print(f'COCO标注已保存到: {output_json}')

yolo_to_coco(LABEL_DIR, IMG_DIR, CLASS_NAMES, COCO_ANN_PATH)

# 2. 推理并保存为COCO格式预测
model = YOLO(MODEL_PATH).to('cuda:2')
results = []
img_id = 1
for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(IMG_DIR, img_name)
    img = Image.open(img_path)
    width, height = img.size
    pred = model(img_path)[0]
    boxes = pred.boxes.cpu().numpy()
    for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        results.append({
            "image_id": img_id,
            "category_id": int(cls) + 1,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score)
        })
    img_id += 1
with open(COCO_RES_PATH, 'w') as f:
    json.dump(results, f)
print(f'预测结果已保存到: {COCO_RES_PATH}')

# 3. 用pycocotools评估
cocoGt = COCO(COCO_ANN_PATH)
cocoDt = cocoGt.loadRes(COCO_RES_PATH)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
