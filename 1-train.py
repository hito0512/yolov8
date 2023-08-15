from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# Use the model
model.train(data="classes.yaml", epochs=200)  # ultralytics\datasets\classes.yaml
path = model.export(format="onnx")  # export the model to ONNX format