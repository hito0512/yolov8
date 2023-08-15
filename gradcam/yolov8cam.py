'''
# -*- encoding: utf-8 -*-
# 说明    : yolov8 cam
# 时间    : 2023/07/31 16:15:51
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3
'''

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
import math
from gradcam.ModelEdit import ModelEdit
from utils import read_image, nms, color_print, correct_boxes, softmax,sigmoid, timer
import time

# yolov8
class GradCam(object):
    r""" 输出层的名字如果为空则选择所有的输出，如果不为空，则选择指定的输出
    Arg:   """
    def __init__(self, model_name, split_layer_name, output_layer_name = [], score_thresh=0.6, nms_thresh=0.5):
        self.modeledit = ModelEdit(model_name,split_layer_name)
        self.output_layer_name = [] if output_layer_name=='' else output_layer_name
        self.module_name  = 'yolov8 cam'
        self.score_thresh = score_thresh
        self.nms_thresh   = nms_thresh
    
    def __call__(self, img_input, img_shape):
        # 原始模型输入的尺寸
        model_shape = self.modeledit.model_input
        # 将模型分为两段
        model_split_befor, model_split_after = self.modeledit(self.output_layer_name)
        all_grades = []
        all_split_conv_output  = []
        all_boxes = []
        # a = time.time()
        with tf.GradientTape() as tape:
            last_conv_layer_output = model_split_befor(img_input)
            tape.watch(last_conv_layer_output)
            preds = model_split_after(last_conv_layer_output)
            
            for i in range(len(preds)):
                logits = []
                boxes = []
                feat_h,feat_w,feat_c = preds[i].shape[1:]
                # 这里不能用numpy进行操作
                feat = tf.reshape(preds[i], (feat_h*feat_w, feat_c))
                scale_x = model_shape[1]/feat_w
                scale_y = model_shape[0]/feat_h
                # 生成anchors
                sx, sy = np.meshgrid(range(0,feat_h), range(0,feat_w))
                sx = sx.flatten() + 0.5
                sy = sy.flatten() + 0.5
                anchors = np.stack((sx, sy))
                anchors = anchors.T
                feat_box = feat[:,:64]
                feat_cls = feat[:,64:]
                # 对box再做处理
                feat_box = np.reshape(feat_box,newshape=(feat_h*feat_w,4,-1))
                box_h, box_w, _ = feat_box.shape
                for j in range(box_h):
                    # cls = K.sigmoid(feat_cls[j])
                    cls = sigmoid(feat_cls[j])
                    
                    # 找到最大得分的位置，这里必须用tensor表示，numpy的话无法进行梯度回传，只需对得分进行该操作，bbox仍然可以继续用numpy操作
                    # obj_index = tf.argmax(cls)  # 不能用np.argmax(cls)
                    obj_index = np.argmax(cls)
                    score = cls[obj_index]
                    if score < self.score_thresh:
                        continue
                    
                    # 记录得分的行列索引
                    local_index = [j, obj_index]  # [row, col]
                    logits.append(local_index)
                    dbox = []
                    for k in range(box_w):
                        channel_val = feat_box[j,k,:]
                        soft_channel = softmax(channel_val)
                        val = self.conv(soft_channel)
                        dbox.append(val)
                    x1y1 = (anchors[j] - dbox[:2])*scale_x
                    x2y2 = (anchors[j] + dbox[2:])*scale_y
                    
                    xmin,ymin,xmax,ymax=correct_boxes(x1y1[0],x1y1[1],x2y2[0],x2y2[1], model_shape, img_shape)
                    xmin=math.floor(xmin+0.5)
                    ymin=math.floor(ymin+0.5)
                    xmax=math.floor(xmax+0.5)
                    ymax=math.floor(ymax+0.5)
                    box=[score,obj_index,xmin,ymin,xmax,ymax]
                    boxes.append(box)
                
                boxes, new_logit = nms(boxes,self.nms_thresh, logits)
                if len(boxes):
                    top_class_channel = feat_cls[new_logit[0][0],new_logit[0][1]]
                    # print('\n',str(time.time()-a))
                    # 反向求取梯度
                    grads = tape.gradient(top_class_channel, last_conv_layer_output)
                    all_grades.append(grads)
                    all_boxes.append(boxes)
                    all_split_conv_output.append(last_conv_layer_output)
                else:
                    all_grades.append(None)
                    all_boxes.append(None)
                    all_split_conv_output.append(None)
        return all_grades, all_boxes, all_split_conv_output

    def specify_output(self, img_input, img_shape):
        # 原始模型输入的尺寸
        model_shape = self.modeledit.model_input
        # 将模型分为两段
        model_split_befor, model_split_after = self.modeledit(self.output_layer_name)
        logits = []
        boxes = []
        a = time.time()
        with tf.GradientTape() as tape:
            last_conv_layer_output = model_split_befor(img_input)
            tape.watch(last_conv_layer_output)
            preds = model_split_after(last_conv_layer_output)
            
            # preds = preds[0]
            feat_h,feat_w,feat_c = preds.shape[1:]
            # 这里不能用numpy进行操作
            feat = tf.reshape(preds, (feat_h*feat_w, feat_c))
            scale_x = model_shape[1]/feat_w
            scale_y = model_shape[0]/feat_h
            # 生成anchors
            sx, sy = np.meshgrid(range(0,feat_h), range(0,feat_w))
            sx = sx.flatten() + 0.5
            sy = sy.flatten() + 0.5
            anchors = np.stack((sx, sy))
            anchors = anchors.T
            feat_box = feat[:,:64]
            feat_cls = feat[:,64:]
            # 对box再做处理
            feat_box = np.reshape(feat_box,newshape=(feat_h*feat_w,4,-1))
            box_h, box_w, _ = feat_box.shape
            for j in range(box_h):
                # cls = K.sigmoid(feat_cls[j])
                cls = sigmoid(feat_cls[j])
                
                # 找到最大得分的位置，这里必须用tensor表示，numpy的话无法进行梯度回传，只需对得分进行该操作，bbox仍然可以继续用numpy操作
                # obj_index = tf.argmax(cls)  # 不能用np.argmax(cls)
                obj_index = np.argmax(cls)
                score = cls[obj_index]
                if score < self.score_thresh:
                    continue
                
                # 记录得分的行列索引
                local_index = [j, obj_index]  # [row, col]
                logits.append(local_index)
                dbox = []
                for k in range(box_w):
                    channel_val = feat_box[j,k,:]
                    soft_channel = softmax(channel_val)
                    val = self.conv(soft_channel)
                    dbox.append(val)
                x1y1 = (anchors[j] - dbox[:2])*scale_x
                x2y2 = (anchors[j] + dbox[2:])*scale_y
                
                xmin,ymin,xmax,ymax=correct_boxes(x1y1[0],x1y1[1],x2y2[0],x2y2[1], model_shape, img_shape)
                xmin=math.floor(xmin+0.5)
                ymin=math.floor(ymin+0.5)
                xmax=math.floor(xmax+0.5)
                ymax=math.floor(ymax+0.5)
                box=[score,obj_index,xmin,ymin,xmax,ymax]
                boxes.append(box)
            boxes, new_logit = nms(boxes,self.nms_thresh, logits)
            top_class_channel = feat_cls[new_logit[0][0],new_logit[0][1]]
        print('\n',str(time.time()-a))
        # 反向求取梯度
        grads = tape.gradient(top_class_channel, last_conv_layer_output)
        return grads, last_conv_layer_output
    
    def __str__(self):
        return self.module_name

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def conv(x):
        w = np.array(range(16))
        x1 = x * w
        return np.sum(x1)


