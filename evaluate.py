from generator import Generator
from yolo_predict import YOLO
import numpy as np
from config import config


yolo = YOLO("./logs_2/class_21-100.h5")


with open(config.VAL_TEXT) as f:
        lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines))

gen = Generator(val_lines=lines)
ap = yolo.evaluate(gen, num_val)
print(ap)