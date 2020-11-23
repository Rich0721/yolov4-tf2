from utils.generator import DataGenerator
from utils.yolov4 import Yolov4
from config import Config

data_gen_train = DataGenerator(Config['annotation_folder'], Config['image_folder'], Config['train_text'])
data_gen_val = DataGenerator(Config['annotation_folder'], Config['image_folder'], Config['val_text'], data_augments=False)

model = Yolov4(config=Config)
model.fit(train_data_gen=data_gen_train,val_data_gen=data_gen_val)