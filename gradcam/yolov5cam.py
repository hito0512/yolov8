'''
# -*- encoding: utf-8 -*-
# 说明    : yolov5 cam
# 时间    : 2023/08/07 09:37:41
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
from utils import color_print


class GradCam(object):
    def __init__(self, ):
        
        pass
    
    def __call__(self, x):
        
        return x
    
    
    