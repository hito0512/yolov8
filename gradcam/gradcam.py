'''
# -*- encoding: utf-8 -*-
# 说明    : 使用CAM方法对输出特征进行可视化
# 时间    : 2023/08/07 09:24:32
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3
'''
import argparse
import tensorflow as tf
import cv2
import numpy as np
import importlib
from utils import color_print, read_image, save_folder,get_filelist,draw_bbox,get_classes
from enum import Enum,unique
from pathlib import Path
from tqdm import tqdm

@unique
class DetecType(Enum):
    yolov2 = 0
    yolov3 = 1
    yolov4 = 2
    yolov5 = 3
    yolox  = 4
    yolov7 = 5
    yolov8 = 6


class ModelGradCam(object):
    r""" 输出层的名字如果为空则选择所有的输出，如果不为空，则选择指定的输出
    Arg:   """
    def __init__(self, model_type, model_name, split_layer_name, output_layer_name = [], class_file='', score_thresh=0.6, nms_thresh=0.5):
        self.save_path = save_folder(model_name)
        GradCam = importlib.import_module('{}cam'.format(model_type))
        self.CAM = GradCam.GradCam(model_name, split_layer_name, output_layer_name, score_thresh, nms_thresh)
        self.classes  = get_classes(class_file)
        
        
        color_print('model: '+self.CAM.module_name)
        
    
    def __call__(self, img_name):
        ori_img ,img_input, img_shape = read_image(img_name, self.CAM.modeledit.model_input)
        # 根据不同的模型求取梯度图像
        self.grads, boxes, self.last_conv_layer_output = self.CAM(img_input, img_shape)
        for i, (grad, last_conv) in enumerate(zip(self.grads, self.last_conv_layer_output)):
            if grad != None:
                ori_img = draw_bbox(ori_img,boxes,classes=self.classes)
                # 对梯度图进行处理，并保存
                img_name = Path(img_name).stem+ '_' + str(i) + '_' + Path(img_name).suffix
                # self.grads_cam(ori_img ,img_shape, img_name, grad, last_conv)
                self.guided_gradcam(ori_img, img_shape, img_name, grad, last_conv)
        
    def guided_gradcam(self, ori_img, img_shape, img_name, grad, last_conv):
        grad = grad[0]
        last_conv = last_conv[0]
        guided_grads = (tf.cast(last_conv > 0, "float32") * tf.cast(grad > 0, "float32") * grad)
        # 对梯度按通道进行求平均
        pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
        guided_gradcam = np.ones(last_conv.shape[:2], dtype=np.float32)
        # last_conv_layer_output = last_conv.numpy()[0]
        # pooled_grads = pooled_grads.numpy()
        # 对梯度图进行加权
        # for i in range(pooled_grads.shape[-1]):
        #     last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
        for i, w in enumerate(pooled_guided_grads):
            guided_gradcam += w * last_conv[:, :, i]
    
        # gradcam = np.mean(last_conv_layer_output, axis=-1)
        gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        gradcam = cv2.resize(gradcam, img_shape[:-1][::-1])
        gradcam = np.uint8(gradcam *255)

        heatmap = cv2.cvtColor(gradcam, cv2.COLOR_GRAY2RGB)
        heatmap =  cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        n_heatmat = (heatmap / 255).astype(np.float32)
        res_img = ori_img / 255
        # res_img = cv2.add(res_img, n_heatmat)
        res_img = res_img+ 0.7*n_heatmat
        res_img = (res_img / res_img.max())
            
        cam = np.uint8(255 * res_img)

        cv2.imwrite(f'{self.save_path}/{img_name}', cam)
    
    def grads_cam(self, ori_img, img_shape, img_name, grad, last_conv):
        # 对梯度按通道进行求平均
        pooled_grads = tf.reduce_mean(grad, axis=(0, 1, 2))
        last_conv_layer_output = last_conv.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        # 对梯度图进行加权
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]
        # 再次按通道求均值
        gradcam = np.mean(last_conv_layer_output, axis=-1)
        gradcam = np.clip(gradcam, 0, np.max(gradcam))
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
        gradcam = cv2.resize(gradcam, img_shape[:-1][::-1])
        gradcam = np.uint8(gradcam *255)

        heatmap = cv2.cvtColor(gradcam, cv2.COLOR_GRAY2RGB)
        heatmap =  cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        n_heatmat = (heatmap / 255).astype(np.float32)
        res_img = ori_img / 255
        # res_img = cv2.add(res_img, n_heatmat)
        res_img = res_img+ 0.7*n_heatmat
        res_img = (res_img / res_img.max())
            
        cam = np.uint8(255 * res_img)

        cv2.imwrite(f'{self.save_path}/{img_name}', cam)
        
        
def calGradCam(model_type, model_name, split_layer_name, output_layer_name,classes_file, score_thresh=0.6, nms_thresh=0.5):
    assert model_type in [d.name for d in DetecType], color_print('the detecter of %s is not support!'%(model_type))
    gradcam = ModelGradCam(model_type, model_name, split_layer_name, output_layer_name, classes_file, score_thresh, nms_thresh)
    files = get_filelist('cam_img')
    for file in tqdm(files[:]):
        gradcam(file)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='目标检测模型推理程序')
    parser.add_argument('-model_type',          type=str,   default="yolov8", \
                        choices=[d.name for d in DetecType],help="模型类型,可以是yolov2,yolov3,yolov4,yolov5")
    parser.add_argument('-model_name',          type=str,   default='./weights/yolov8n.h5',             help="模型文件") # yolov5s_prune_0
    parser.add_argument('-classes_file',        type=str,   default="../../dataset/classes.txt",        help="保存类别名称的文件")
    parser.add_argument('-anchors_file',        type=str,   default="../../dataset/anchors_v3v4v5.txt", help="保存anchors的文件")
    
    parser.add_argument('-split_layer_name',    type=str,   default='254', choices=['254','269','284'], help="要分段的层名字")
    parser.add_argument('-output_layer_name',   type=str,   default="output0", choices=['output0','307','318',''], help="模型的输出")
    
    parser.add_argument('-score_thresh',        type=float, default=0.6,        help="置信度阈值")
    parser.add_argument('-nms_thresh',          type=float, default=0.5,        help="NMS阈值")
    parser.add_argument('-prune',               type=int,   default=0, choices=[0,1] ,help="是否是yolov5剪枝, 0:不是,1:是")
    args = parser.parse_args()
    
    calGradCam(args.model_type, args.model_name, args.split_layer_name, args.output_layer_name, args.classes_file, args.score_thresh, args.nms_thresh)