from ModelEdit import ModelEdit
from tensorflow.keras.layers import Input, ReLU,Concatenate




model_name = './weights/yolov8n.h5'

# ---------------- replace
# layer_name = '285'
# job = 'replace'
# new_layer = ReLU()

# me = ModelEdit(model_name,job=job)
# me(layer_name, new_layer)


# ---------------- split
# job = 'split'
# layer_name = '254'
# output = '316'

# me = ModelEdit(model_name,job=job)
# me(layer_name)

# ---------------- delete
# job = 'delete'
# layer_name = '316'
# output = '316'

# me = ModelEdit(model_name,job=job)
# me(layer_name)

# ---------------- insert
job = 'insert'
layer_name = '285'
add_layer = ReLU()

me = ModelEdit(model_name,job=job)
me(layer_name, add_layer)
