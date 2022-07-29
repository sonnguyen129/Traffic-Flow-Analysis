import sys
sys.path.insert(0, './YOLOX')
import torch
import cv2
from yolox.utils import vis
import time
from yolox.exp import get_exp
import numpy as np
from collections import deque
from collections import Counter

# importing Detector
from yolox.data.datasets.coco_classes import COCO_CLASSES
from detector import Predictor

# Importing Deepsort
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

# Importing Visuals
from visuals import *

from intersect_ import *

import math

import datetime

# A Dictionary to keep data of tracking
data_deque = {}
speed_dict = {}

class_names = COCO_CLASSES

lines  = [
    {'Title' : 'North', 'Cords' : [(1720, 561), (1111, 505)]},
    {'Title' : 'South', 'Cords' : [(625, 727), (1532, 861)]},
    {'Title' : 'East', 'Cords' : [(1764, 595), (1731, 806)]},
    {'Title' : 'West', 'Cords' : [(905, 515), (586, 657)]}
]

object_counter = {
    'North' : Counter(),
    'South' : Counter(),
    'East'  : Counter(),
    'West'  : Counter(),
}


pts = {}

def vis_track(img, outputs):
    if len(outputs) == 0:
        return img

    for key in list(pts):
        if key not in outputs[:,-2]:
            pts.pop(key)

    for i in range(len(outputs)):
        box = outputs[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        id = box[4]
        clsid = box[5]

        if id not in pts:
            pts[id] = deque(maxlen=64)

        # pts = { '1' : deque(),'2' : deque()}

        center = (int((x0+x1)/2) , int((y0+y1)/2))
        pts[id].append(center)

        # Drawing a circle
        color = compute_color_for_labels(clsid)
        thickness = 5
        cv2.circle(img,  (center), 1, color, thickness)

        # Draw motion path
        for j in range(1, len(pts[id])):
            if pts[id][j - 1] is None or pts[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 3)
            cv2.line(img,(pts[id][j-1]), (pts[id][j]),(color),thickness)

    return img



def estimateSpeed(location1, location2):

    height = location1[0] - location2[0]
    width = location1[1] - location2[1]
    
    distance_in_pixels = math.sqrt(math.pow(height,2) + math.pow(width,2))

    pixels_per_meter = 15

    distance_in_meters = distance_in_pixels/pixels_per_meter

    fps = 30 
    Time_  = 1/fps

    speed_mps = distance_in_meters/Time_

    speed_kmph = speed_mps*(3600/1000)

    return int(speed_kmph)





#Draw the Lines
def draw_lines(lines, img):
    for line in lines:
        img = cv2.line(img, line['Cords'][0], line['Cords'][1], (255,255,255), 3)
    return img

# Update the Counter
def update_counter(centerpoints, obj_name, id):
    data = []
    for line in lines:
        p1 = Point(*centerpoints[0])
        q1 = Point(*centerpoints[1])
        p2 = Point(*line['Cords'][0])
        q2 = Point(*line['Cords'][1])
        if doIntersect(p1, q1, p2, q2):
            object_counter[line['Title']].update([obj_name])
            speed = estimateSpeed(location1 = centerpoints[0], location2 = centerpoints[1])
            speed_dict[id] = speed
            print("intersection detected")
            data.append({
                'Category' : obj_name,
                'direction': line['Title'],
                'Time'     : datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Speed'    : speed,
                'id'       : id
                    }) 
    return data

# Draw the Final Results
def draw_results(img):
    x = 100
    y = 100
    offset = 50
    for line_name, line_counter in object_counter.items():
        Text = line_name + " : " + ' '.join([f"{label}={count}" for label, count in line_counter.items()])
        cv2.putText(img, Text, (x,y), 6, 1, (104, 52, 235), 3, cv2.LINE_AA)
        y = y+offset
    return img



# Function to calculate delta time for FPS when using cuda
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# Draw the boxes having tracking indentities 
def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape 
    # Cleaning any previous Enteries
    [data_deque.pop(key) for key in set(data_deque) if key not in identities]
    [speed_dict.pop(key) for key in set(data_deque) if key not in identities]
    
    frame_data = []
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) +offset[0]  for i in box]  
        box_height = (y2-y1)
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in set(data_deque):  
          data_deque[id] = deque(maxlen= 100)

        color = compute_color_for_labels(object_id[i])
        obj_name = class_names[object_id[i]]
        label = '%s' % (obj_name)
        
        data_deque[id].appendleft(center) #appending left to speed up the check we will check the latest map

        if len(data_deque[id]) >=2:
            data = update_counter(centerpoints = data_deque[id], obj_name = obj_name, id = id)
            frame_data.extend(data)

        if id in speed_dict:
            speed = speed_dict[id]
        else:
            speed = ''
        
        UI_box(box, img, label=label + str(speed) + 'km/h', color=color, line_thickness=3, boundingbox=True)
        

    return img, frame_data

# Tracking class to integrate Deepsort tracking with our detector
class Tracker():
    def __init__(self, filter_classes=None, model='yolox-s', ckpt='wieghts/yolox_s.pth'):
        self.detector = Predictor(model, ckpt)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        self.filter_classes = filter_classes
    def update(self, image, visual = True, logger_=True):
        height, width, _ = image.shape 
        _,info = self.detector.inference(image, visual=False, logger_=logger_)
        outputs = []
        
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            objectids = []
            for [x1, y1, x2, y2], class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_classes:
                    if class_names[class_id] not in set(filter_classes):
                        continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])  
                objectids.append(info['class_ids'])             
                scores.append(score)
                
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.deepsort.update(bbox_xywh, scores, info['class_ids'],image)
            data = []
            if len(outputs) > 0:
                bbox_xyxy =outputs[:, :4]
                identities =outputs[:, -2]
                object_id =outputs[:, -1]
                image, frame_data = draw_boxes(image, bbox_xyxy, object_id,identities)
            else:
                frame_data = []

            return image, outputs, frame_data

