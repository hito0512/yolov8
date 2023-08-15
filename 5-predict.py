from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# metrics = model.val()  # 测试所有的验证集
results = model("./11_17_50_534.png")  # predict on an image

print(results)
