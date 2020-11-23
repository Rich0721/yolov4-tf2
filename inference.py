from tensorflow.keras import models
from utils.yolov4 import Yolov4

model = Yolov4(weight_path="./logs/yolov4-39.h5")
model.predict("../datasets/voc/JPEGImages/00167.jpg", random_color=False)