'''
# -*- encoding: utf-8 -*-
# 说明    : 
# 时间    : 2023/08/04 17:59:08
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3 or pytorch1.7 or TensorFlow1.15 or vitis1.3
'''
import random
import time
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from copy import deepcopy
import colorsys

def color_print(msg):
    no_color = "\033[0m"
    all_color = ['\033[0;%sm'%(str(i)) for i in range(31,38)]
    random_color = random.choice(all_color)
    print('%s%s%s'%(random_color, str(msg), no_color))

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        color_print(f"{func.__name__} eslaped {end-start:.2f} seconds" )
        return result
    return wrapper

def letterbox_image(image, new_shape):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw,ic= image.shape
    h ,w,_= new_shape
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)

    new_image = np.ones((h,w,ic), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image

    return new_image

def read_image(path, model_shape):
    img = cv2.imread(path)
    img_input = letterbox_image(img, model_shape)
    img_input = img_input/255.0
    if model_shape[2] == 1:
        img_input = img_input[:,:,0:1]
    else:
        img_input = img_input[:,:,::-1]
    img_input = np.expand_dims(img_input,axis=0)
    
    return img.astype(np.float32), img_input, img.shape

def save_folder(model_name):
    path = 'cam_'+Path(model_name).stem
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

# 经过测试模型输入大小与测试图片大小相同或者不同，修正后的box都是正确的。
def correct_boxes(xmin, ymin, xmax, ymax, input_shape, image_shape):
    new_w,new_h=0,0
    img_h,img_w,_=image_shape
    input_h,input_w,_=input_shape
    if input_w/img_w <input_h/img_h:
        new_w=input_w
        new_h=img_h*input_w/img_w
    else:
        new_w=img_w*input_h/img_h
        new_h=input_h
    cx=(xmin+xmax)/2
    cy=(ymin+ymax)/2
    cx=(cx-(input_w-new_w)/2)/(new_w/img_w)
    cy=(cy-(input_h-new_h)/2)/(new_h/img_h)
    w=xmax-xmin
    h=ymax-ymin
    w=w*img_w/new_w
    h=h*img_h/new_h

    xmin = max(0, cx-0.5*w)
    ymin = max(0, cy-0.5*h)
    xmax = min(cx+0.5*w, img_w)
    ymax = min(cy+0.5*h, img_h)

    box=[xmin, ymin, xmax, ymax]
    return box

def get_classes(filename):
    classes=[]
    with open(filename,"rt")as f:
        for line in f:
            cls=line.strip()
            classes.append(cls)
    return classes

def draw_bbox(image, bboxes, classes=["ship"], show_label=True):
    """
    bboxes: [probability,cls_id,x_min, y_min, x_max, y_max] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w= image.shape[0:2]
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    # image=cv2.cvtColor(image,cv2.CV_8U)
    for i, bbox in enumerate(bboxes[0]):
        coor = np.array(bbox[2:], dtype=np.int32)
        fontScale = 0.5
        score = bbox[0]
        class_ind = int(bbox[1])
        # print('class_ind ', class_ind)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        # print(' coor[1] ',  coor[1])
        if coor[1] < 10:
            coor[1] += 20
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, 2)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
    return image

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[0], b2[1], b2[2], b2[3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)

    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)

    return iou

def nms(bboxes,iou_thresh, logits):
    ori_box = deepcopy(bboxes)
    bboxes.sort(reverse=True)
    sort_index = []
    for b in bboxes:
        id = ori_box.index(b)
        sort_index.append(id)
    new_logit = [logits[id] for id in sort_index]
    
    bboxes=bboxes[0:200]
    if len(bboxes)>0:
        results=[bboxes[0]]
        nms_logit=[new_logit[0]]
    else:
        return [], []
    for i in range(1,len(bboxes)):
        b1=bboxes[i][2:]
        discard=False
        for j in range(0,len(results)):
            b2=results[j][2:]
            IOU=iou(b1,b2)
            if IOU>=iou_thresh:
                discard=True
        if discard==False:
            results.append(bboxes[i])
            nms_logit.append(new_logit[i])
    return results, nms_logit

def get_filelist(path):
    list_path = os.listdir(path)
    full_path = map(lambda x: path + '/'+ x, list_path)
    
    return list(full_path)


def softmax(x):
    exp = np.exp(x)
    val = exp/np.sum(exp)
    return val


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))