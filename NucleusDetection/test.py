# -*- coding: utf-8 -*-
import csv
w = 80
THRESHOLD = 0.59
file_train = "../VOC2007_old/2007_train.txt"
val_set = ['000024', '000057']

def is_val(line_in_file):
    img_path = line_in_file.split()[0]
    img_name = img_path.split('/')[-1].split('.')[0]
    return (img_name in val_set)

def dist(point1, point2):
    dist = 0
    for (x1, x2) in zip(point1, point2):
        dist += (x1 - x2)**2
    return dist**0.5

def train_set():
    train_set = []
    with open(file_train, 'r') as f:
        for line in f.readlines():
            img_path = line.split()[0]
            img_name = img_path.split('/')[-1].split('.')[0]
            if img_name not in val_set:
                train_set.append(img_name)
    return train_set

train_set = train_set()

def read_data(data_set='val'):
    
    boxes = []
    
    if data_set == 'val':
        with open(file_train, 'r') as f:
            for line in f.readlines(): 
                if is_val(line):
                    boxes.extend(line.split()[1:])    # extract the bounding boxes
    elif data_set == 'train':
        with open(file_train, 'r') as f:
            for line in f.readlines(): 
                if not is_val(line):
                    boxes.extend(line.split()[1:])    # extract the bounding boxes
                
    # ground truth boxes
    boxes = [list(map(int, box.split(',')[:-1])) for box in boxes]   # convert to int
    # ground truth positions
    pos = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
    
    # read results from csv file
    pred_pos = []
    
    for img_name in val_set:
        with open('./Results/Results_' + img_name + '.csv') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            _pred_pos = [[float(row[-2]), float(row[-1])] for row in f_csv]
            pred_pos.extend(_pred_pos)
    
    return pred_pos, pos

pred_pos, pos = read_data(data_set='val')

#%% 
# TP: for each detected nucleus, 
# the center of the closest ground truth nucleus is inside the detected window (80, 80)

def evaluate(pred_pos, pos):
    
    tp = 0
    tp_fp = len(pred_pos)
    tp_fn = len(pos)
    
# =============================================================================
#     # lists for finding failed/false detections
#     preds_tp = []
#     detected_gt = []
# =============================================================================
    
    for pred_p in pred_pos:     # the center of the detected nucleus
        # compute distance list between the ground truth nuclei
        distance_to_gt = [dist([p[0],p[1]], [pred_p[0],pred_p[1]]) for p in pos] 
        # the closest ground truth
        i_closest_gt = distance_to_gt.index(min(distance_to_gt))
        # compute distancce list between the predictions and the closest ground truth
        distance_to_pred = [dist([pred[0],pred[1]], [pos[i_closest_gt][0],pos[i_closest_gt][1]]) for pred in pred_pos]
        # the clossest prediction to that ground truth
        i_closest_pred = distance_to_pred.index(min(distance_to_pred))
        # if not match
        if (pred_p != pred_pos[i_closest_pred]):
            continue
        # if the cloest one is inside the detected window
        if (pred_p[0] - w/2 <= pos[i_closest_gt][0] <= pred_p[0] + w/2 and 
            pred_p[1] - w/2 <= pos[i_closest_gt][1] <= pred_p[1] + w/2):
            tp += 1
            
# =============================================================================
#             # lists for finding failed/false detections
#             preds_tp.append(pred_p)
#             detected_gt.append(pos[i_closest_gt])
#     
#     # 求差集，在前者中但不在后者中
#     fp = [list(map(int, pred_p)) for pred_p in pred_pos if pred_p not in preds_tp]
#     fn = [list(map(int, p)) for p in pos if p not in detected_gt]
# =============================================================================

    precision = tp / tp_fp
    recall = tp / tp_fn
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1#, fp, fn



#%%
#precision, recall, f1, fp, fn = evaluate(pred_pos, pos)
precision, recall, f1 = evaluate(pred_pos, pos)
print("Threshold = \t{}\nPrecision = \t{}\nRecall = \t{}\nF1-score = \t{}"
      .format(THRESHOLD, precision, recall, f1))
print("{}\t{}\t{}\t{}"
      .format(THRESHOLD, precision, recall, f1))


#with open("./Results/ThreshldOpt.csv", "a", newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow([THRESHOLD, precision, recall, f1])
#%%
#fileHeader = ["Threshold", "Precision", "Recall", "F1-score"]
#with open("./Results/ThreshldOpt.csv", "w") as f:
#    writer = csv.writer(f)
#    writer.writerow(fileHeader)

#%% find failed/false 
# =============================================================================
# import cv2
# 
# for img_name in val_set:
#     img = cv2.imread('../VOC2007/JPEGImages/' + img_name + '.jpg')
#     
#     i = 0
#     for center in fn:
#         box = [center[0] - int(w/2), center[1] - int(w/2), center[0] + int(w/2), center[1] + int(w/2)]
#         cropped = img[box[1]:box[1]+w, box[0]:box[0]+w]    # [Ymin:Ymax , Xmin:Xmax]
#         cv2.imwrite("../VOC2007/Patches/Prediction/FN/fn_{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i)), 
#                     cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#         i += 1
#         
#     j = 0
#     for center in fp:
#         box = [center[0] - int(w/2), center[1] - int(w/2), center[0] + int(w/2), center[1] + int(w/2)]
#         cropped = img[box[1]:box[1]+w, box[0]:box[0]+w]    # [Ymin:Ymax , Xmin:Xmax]
#         cv2.imwrite("../VOC2007/Patches/Prediction/FP/fn_{}_{}.jpg".format(img_name, '0'*(4 - len(str(j))) + str(j)), 
#                     cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#         j += 1
# =============================================================================
