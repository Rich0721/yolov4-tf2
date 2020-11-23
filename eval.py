from utils.generator import DataGenerator
from utils.yolov4 import Yolov4
from config import Config

model = Yolov4(weight_path="./logs/yolov4-50.h5")
data_gen_val = DataGenerator(Config['annotation_folder'], Config['image_folder'], Config['test_text'])


print(model.evaluate(data_gen_val))
