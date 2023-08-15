'''
# -*- encoding: utf-8 -*-
# 文件    : start_convert.py
# 说明    : onnx 模型转换为keras
# 时间    : 2022/06/24 11:25:17
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3 
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import onnx
from onnx2keras import onnx_to_keras
# from ..onnx2keras 
model_name = './weights/yolov8n'
# Load ONNX model
onnx_model = onnx.load(model_name + '.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['images'],change_last=True)
k_model.save(model_name + ".h5")


