'''
# -*- encoding: utf-8 -*-
# 说明    : 节点
# 时间    : 2023/07/31 16:15:51
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3
'''

import numpy as np
import cv2
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from pathlib import Path
import os
import tensorflow as tf
import shutil
import random
import time
from pathlib import Path
import math
from copy import deepcopy

def color_print(msg):
    no_color = "\033[0m"
    all_color = ['\033[0;%sm'%(str(i)) for i in range(31,38)]
    random_color = random.choice(all_color)
    print('%s%s%s'%(random_color, str(msg), no_color))

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

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(x):
    exp = np.exp(x)
    val = exp/np.sum(exp)
    return val

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

def read_image(path):
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

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        color_print(f"{func.__name__} eslaped {end-start:.2f} seconds" )
        return result
    return wrapper

def conv(x):
    w = np.array(range(16))
    x1 = x * w
    return K.sum(x1)

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

class TensorDict(dict):
    def __init__(self):
        super().__init__()

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key.ref(), value)
        except AttributeError:
            super().__setitem__(key.experimental_ref(), value)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item.ref())
        except AttributeError:
            return super().__getitem__(item.experimental_ref())


class ModelEdit(object):
    def __init__(self, model_name, layer_name):
        self.model_name = model_name
        self.layer_name = layer_name
        self.model = load_model(model_name)
        self.nodes = []
        self._finished_nodes = {}
        self._new_input = {}
    
    @timer
    def __call__(self, output_name=[]):
        r""" 如果为空则选择所有的输出，如果不为空，则选择指定的输出
        Arg:   """
        # 找到要分段节点的父节点和本身
        parent_nodes, node = self.find_node()
        # 保存前半段模型
        front_model = self.front_model(parent_nodes)
        # 重建后半段模型
        output_nodes = []
        for idx, output in enumerate(self.model.outputs):
            layer_name = output.name
            name = layer_name.split('/')[0]
            layer, node_index, tensor_index = output._keras_history
            if len(output_name) == 0:
                output_nodes.append(layer.inbound_nodes[node_index])
            else: 
                 if name in output_name:
                     output_nodes.append(layer.inbound_nodes[node_index])
                     
        new_outputs = self._rebuild_graph([parent_nodes.output_tensors], output_nodes)
        
        # 分段的后部分模型
        back_model = Model(list(self._new_input.values()), new_outputs)
        # back_model.save(Path(self.model_name).stem + '_split_'+self.layer_name + '_after.h5')
        
        return front_model, back_model
    
    def _rebuild_graph(self, graph_inputs, output_nodes, graph_input_masks=None):
        r""" 
        这里采用倒推的方式，根据指定的最后一个输出，反方向找到要分段节点的上一个节点
        Arg:   """
        if not graph_input_masks:
            graph_input_masks = [None] * len(graph_inputs)
            
        def _rebuild_rec(node):
            layer = node.outbound_layer

            # color_print('[getting inputs for] : '+layer.name)
            node_output = self.single_element(node.output_tensors)
            
            # 节省时间，否则会重复递归寻找，耗时巨大！！！
            if node in self._finished_nodes.keys():
                return self._finished_nodes[node]
                
            mask_map = TensorDict()
            # 以字典的方式记录当前节点的位置
            for input, mask in zip(graph_inputs, graph_input_masks):
                mask_map[input] = mask
                
            try:
                # 反向递归，直到父节点的输出是当前节点
                output_mask = mask_map[node_output]
                # 把要分段节点的父节点直接改为新的输入
                new_input = Input(shape=node_output.shape[1:])
                # 保存新的输入，用于构建最终新的模型
                self._new_input[node_output.name] = new_input
                
                return new_input, output_mask
            
            except KeyError:
                inbound_nodes = node.parent_nodes
                inputs, input_masks = zip(*[_rebuild_rec(n) for n in inbound_nodes])
                input_masks = input_masks[0]
                new_layer = node.outbound_layer
                # 这里运行一下，初始化当前层输入和输出shape
                output = new_layer(self.single_element(list(inputs)))
                # color_print('[当前层] : '+new_layer.name)
                color_print('layer complete: {0}'.format(layer.name))
                
                # 已经处理过的层保存起来，后面再找到该层的时候直接使用，节省时间。否则又要重新递归一下，会非常耗时。
                self._finished_nodes[node] = (output, input_masks)
                return output, input_masks

        outputs, output_masks = zip(*[_rebuild_rec(n) for n in output_nodes]) 
        
        return outputs
    
    @staticmethod
    def single_element(x):
        if isinstance(x, tf.Tensor):
            return x
        if len(x) == 1:
            x = x[0]
        return x

    def find_node(self):
        # 从这一层起开始分段
        layer = self.model.get_layer(self.layer_name)
        node = layer.inbound_nodes[0]
        parent_nodes = node.parent_nodes[0]
        
        return parent_nodes, node
    
    def front_model(self, parent_nodes):
        output = parent_nodes.layer.output
        front_model = Model(inputs=self.model.inputs, outputs=output)
        # front_model.save(Path(self.model_name).stem + '_split_'+self.layer_name + '_befor.h5')
        return front_model



get_filelist('cam_img')
model_name = './weights/yolov8n.h5'
img_name = '13_50_17_243.png'
save_path = save_folder(model_name)

model = load_model(model_name)
model_shape = np.array(model.input.shape[1:])
ori_img ,img_input, img_shape = read_image(img_name)

split_layer_name = '254'  # 254       269
output_name = ['output0']     # output0   307
modeledit = ModelEdit(model_name,split_layer_name)
model1, model2 = modeledit(output_name)

logits = []
boxes  = []
score_thresh = 0.6
nms_thresh = 0.5
with tf.GradientTape() as tape:
    last_conv_layer_output = model1(img_input)
    tape.watch(last_conv_layer_output)
    preds = model2(last_conv_layer_output)
    
    preds = preds[0]
    feat_h,feat_w,feat_c = preds.shape[1:]
    # 这里也不能用numpy进行操作
    feat = K.reshape(preds, (feat_h*feat_w, feat_c))
    
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
        cls = K.sigmoid(feat_cls[j])
        
        # 找到最大得分的位置，这里必须用tensor表示，numpy的话无法进行梯度回传，只需对得分进行该操作，bbox仍然可以继续用numpy操作
        obj_index = tf.argmax(cls)  # 不能用np.argmax(cls)
        score = cls[obj_index]
        if score < score_thresh:
            continue
        
        # 记录得分的行列索引
        local_index = [j, obj_index]  # [row, col]
        logits.append(local_index)
        dbox = []
        for k in range(box_w):
            channel_val = feat_box[j,k,:]
            soft_channel = softmax(channel_val)
            val = conv(soft_channel)
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
    boxes, new_logit = nms(boxes,nms_thresh, logits)
    top_class_channel = feat_cls[new_logit[0][0],new_logit[0][1]]
# 反向求取梯度
grads = tape.gradient(top_class_channel, last_conv_layer_output)
# 对梯度按通道进行求平均
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
last_conv_layer_output = last_conv_layer_output.numpy()[0]
pooled_grads = pooled_grads.numpy()
# 对梯度图进行加权
for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]
# 再次按通道求均值
gradcam = np.mean(last_conv_layer_output, axis=-1)
gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
gradcam = cv2.resize(gradcam, img_shape[:-1][::-1])
gradcam = np.uint8(gradcam *255)

heatmap = cv2.cvtColor(gradcam, cv2.COLOR_GRAY2RGB)
heatmap =  cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

n_heatmat = (heatmap / 255).astype(np.float32)
res_img = ori_img / 255
res_img = cv2.add(res_img, n_heatmat)
res_img = (res_img / res_img.max())
      
cam = np.uint8(255 * res_img)


cv2.imwrite(f'{save_path}/{img_name}.jpg', cam)

