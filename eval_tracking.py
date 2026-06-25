import numpy as np

def read_mot_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if frame not in data:
                data[frame] = []
            data[frame].append([obj_id, x, y, x+w, y+h])
    return data

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate(gt_file, pred_file, iou_threshold=0.5):
    gt = read_mot_file(gt_file)
    pred = read_mot_file(pred_file)

    all_frames = set(gt.keys()) | set(pred.keys())

    TP = 0
    FP = 0
    FN = 0
    total_iou = 0
    id_switches = 0
    prev_matches = {}

    for frame in sorted(all_frames):
        gt_boxes = gt.get(frame, [])
        pred_boxes = pred.get(frame, [])

        matched_gt = set()
        matched_pred = set()
        current_matches = {}

        for pi, pb in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou_val = iou(pb[1:], gb[1:])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi != -1:
                TP += 1
                total_iou += best_iou
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                gt_id = gt_boxes[best_gi][0]
                pred_id = pb[0]
                current_matches[gt_id] = pred_id
                if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                    id_switches += 1
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)
        prev_matches = current_matches

    MOTA = 1 - (FP + FN + id_switches) / max(sum(len(v) for v in gt.values()), 1)
    MOTP = total_iou / max(TP, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    IDF1 = 2 * TP / max(2 * TP + FP + FN, 1)

    print("=" * 40)
    print("МЕТРИКИ ТРЕКИНГА ПАРАПЛАНОВ")
    print("=" * 40)
    print(f"MOTA  (точность трекинга): {MOTA*100:.1f}%")
    print(f"MOTP  (точность позиций):  {MOTP*100:.1f}%")
    print(f"IDF1  (качество ID):       {IDF1*100:.1f}%")
    print(f"Precision:                 {precision*100:.1f}%")
    print(f"Recall:                    {recall*100:.1f}%")
    print(f"TP (найдено верно):        {TP}")
    print(f"FP (ложные):               {FP}")
    print(f"FN (пропущено):            {FN}")
    print(f"ID switches (смен ID):     {id_switches}")
    print("=" * 40)

gt_file = r'C:\Users\Madina\Desktop\mot_metrics\gt\gt.txt'
pred_file = r'C:\Users\Madina\PycharmProjects\pythonProject3\YOLOv8-SMOT\results\predictions\7s\internet\tracks\7s.txt'

evaluate(gt_file, pred_file)