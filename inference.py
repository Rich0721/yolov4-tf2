from tensorflow.keras import models
from utils.yolov4 import Yolov4

model = Yolov4(weight_path="./logs/yolov414.h5")
model.predict("../datasets/voc/JPEGImages/00000.jpg", random_color=False)