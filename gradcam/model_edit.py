'''
# -*- encoding: utf-8 -*-
# 说明    : 对模型进行编辑,包括模型分割，节点替换，节点删除，节点插入，模型拼接
# 时间    : 2023/08/04 17:37:14
# 作者    : Hito
# 版本    : 1.0
# 环境    : TensorFlow2.3 
'''


from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from pathlib import Path
import tensorflow as tf
from utils import timer
from pathlib import Path
from utils import color_print
import numpy as np




class ModelEdit(object):
    def __init__(self, model_name, layer_name, save=False):
        self.model_name = model_name
        self.layer_name = layer_name
        self.model = load_model(model_name)
        self.nodes = []
        self._finished_nodes = {}
        self._new_input = {}
        self.save = save
        self.model_input = np.array(self.model.input.shape[1:])
    
    # @timer
    def __call__(self, output_name=[]):
        r""" 如果为空则选择所有的输出，如果不为空，则选择指定的输出
        Arg:   """
        # output_name = output_name.split(',')
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
        
        if self.save:
            name_stem = Path(self.model_name).stem + '_split_'+self.layer_name
            front_model.save(name_stem + '_befor.h5')
            back_model.save( name_stem + '_after.h5')
        
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
        # outputs = outputs[0]
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
        
        return front_model


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





