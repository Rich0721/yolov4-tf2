import xml.etree.ElementTree as ET
from matplotlib.pyplot import box
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import cv2
from tensorflow.python.keras.backend import shape
from config import Config
from imgaug import augmenters as iaa
import imgaug as ia

ia.seed(2)
class DataGenerator(Sequence):

    def __init__(self, annotation_folder, image_folder, text_file, max_boxes=100, shuffle=True, data_augments=True):

        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        self.files = open(text_file).read().split()
        self.num_classes = len(Config["classes"])
        self.classes = Config["classes"]
        self.num_gpu = Config['num_gpu']
        self.batch_size = Config["batch_size"] * Config['num_gpu']
        self.image_size = Config["image_size"]
        self.anchors = np.array(Config['anchors']).reshape((9, 2))
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files))
        self.max_boxes = max_boxes
        self.data_augments = data_augments
        if self.data_augments:
            self.sequential = iaa.Sequential([
                iaa.Flipud(0.5), # vertically flip 20 %
                iaa.Fliplr(0.5), # horizontal flip 20%
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # brightness
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.Affine(translate_px={"x": 15, "y": 15}, scale=(0.8, 0.95), rotate=(-30, 30)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # Add a value of -10 to 10 to each pixel.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),# Same as sharpen, but for an embossing effect
                ]
                ,random_order=True)
            self.sequential = self.sequential.to_deterministic()
        self.on_epoch_end()

    def __len__(self):
        return  int(np.ceil(len(self.files) / self.batch_size))
    
    def __getitem__(self, index):

        idxs = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        #print(idxs)
        input_name = [self.files[i] for i in idxs]
        X, y_tensor, bbox = self.__data_generation(input_name)
        return [X, *y_tensor, bbox], np.zeros(len(input_name))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def size(self):
        return len(self.files)

    def load_image(self, i):
        return cv2.imread(os.path.join(self.image_folder, self.files[i]+".jpg"))
    
    def load_annotations(self, i):
        root = ET.parse(os.path.join(self.annotation_folder, self.files[i] + ".xml"))
        objs = root.findall("object")
        box_data = []
        if len(objs) > 0:
            for obj in objs:
                name = obj.find("name").text
                class_index = self.classes.index(name)
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                box_data.append([xmin, ymin, xmax, ymax, class_index])
        
        if len(box_data) == 0:
            box_data = [[]]

        return np.array(box_data)
        
    def __data_generation(self, input_name):

        X = np.empty((len(input_name), *self.image_size), dtype=np.float32)
        bbox = np.empty((len(input_name), self.max_boxes, 5), dtype=np.float32)

        for i, f in enumerate(input_name):
            img_data, box_data = self.read_image_and_annotation(f)
            X[i] = img_data
            bbox[i] = box_data
        
        y_tensor, y_true_boxes_xywh = preprocess_true_boxes(bbox, self.image_size[:2], self.anchors, self.num_classes)

        return X, y_tensor, y_true_boxes_xywh
            
    def read_image_and_annotation(self, file_name):

        # image process
        image = cv2.imread(os.path.join(self.image_folder, file_name + ".jpg"))
        image = image[:,:, ::-1]
        ih, iw = image.shape[:2]
        h, w, c = self.image_size
        scale_w, scale_h = w/iw, h/ih
        image = cv2.resize(image, (w, h))
        

        # read xml
        root = ET.parse(os.path.join(self.annotation_folder, file_name + ".xml"))
        objs = root.findall("object")
        box_data = []
        if len(objs) > 0:
            for obj in objs:
                name = obj.find("name").text
                class_index = self.classes.index(name)
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find('xmin').text) * scale_w
                ymin = int(bndbox.find('ymin').text) * scale_h
                xmax = int(bndbox.find('xmax').text) * scale_w
                ymax = int(bndbox.find('ymax').text) * scale_h
                box_data.append([xmin, ymin, xmax, ymax, class_index])
            if len(box_data) > self.max_boxes:
                np.random.shuffle(box_data)
                box_data = box_data[:self.max_boxes]

            if self.data_augments and np.random.uniform() < 0.5:
                image_data, box_data = self.data_augmentation(image, box_data)

        image_data = np.array(image) / 255.
        box_data = np.array(box_data)
        
        return image_data, box_data

    def data_augmentation(self, image_data, bbox_data):
        
        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) for box in bbox_data
        ], shape=image_data.shape)
        
        image_data = image_data.astype(np.uint8)
        image_aug = self.sequential.augment_images([image_data])[0]
        
        bbs_augs = self.sequential.augment_bounding_boxes([bbs])[0]
        for i,bbs_aug in enumerate(bbs_augs):
            bbox_data[i][0] = bbs_aug.x1
            bbox_data[i][1] = bbs_aug.y1
            bbox_data[i][2] = bbs_aug.x2
            bbox_data[i][3] = bbs_aug.y2
        
        return image_aug, bbox_data
            
def preprocess_true_boxes(true_boxes, input_shpe, anchors, num_classes):

    '''Preprocess true boxes to training input format'''

    num_stages = 3
    anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    bbox_per_grid = 3
    true_boxes = np.array(true_boxes, dtype='float32')
    true_boxes_abs = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shpe, dtype='int32')
    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2 # center, (100, 2)
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2] # (100, 2)

    # Normalize (0, 1)
    
    true_boxes[..., 0:2] = true_boxes_xy / input_shape[::-1] # xy
    true_boxes[..., 2:4] = true_boxes_wh / input_shape[::-1] #wh

    bs = true_boxes.shape[0]
    grid_sizes = [input_shape // {0:8, 1:16, 2:32}[stage] for stage in range(num_stages)]
    y_true = [np.zeros((bs, grid_sizes[s][0], grid_sizes[s][1], bbox_per_grid, 5+num_classes), dtype='float32') for s in range(num_stages)]
    
    y_true_boxes_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = true_boxes_wh[..., 0] > 0

    for batch_idx in range(bs):
        wh = true_boxes_wh[batch_idx, valid_mask[batch_idx]]# (# of bbox, 2)
        num_boxes = len(wh)
        if num_boxes == 0:
            continue
        wh = np.expand_dims(wh, -2)  # (# of bbox, 1, 2)
        box_maxes = wh / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxes  # (# of bbox, 1, 2) 

        # Compute IoU between each anchors and true boxes for responsibility assignment
        intersect_mins = np.maximum(box_mins, anchor_mins)  # (# of bbox, 9, 2)
        intersect_maxs = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)  # (# of bbox, 9)
        box_area = wh[..., 0] * wh[..., 1]  # (# of bbox, 1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, 9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # (# of bbox, 9)

        # find best anchor for each true box
        best_anchors = np.argmax(iou, axis=-1)

        for box_idx in range(num_boxes):
            best_anchor = best_anchors[box_idx]
            for stage in range(num_stages):
                if best_anchor in anchor_mask[stage]:
                    x_offset = true_boxes[batch_idx, box_idx, 0] * grid_sizes[stage][1]
                    y_offset = true_boxes[batch_idx, box_idx, 1] * grid_sizes[stage][0]
                    # Grid Index
                    grid_col = np.floor(x_offset).astype('int32')
                    grid_row = np.floor(y_offset).astype('int32')
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_idx, box_idx, 4].astype('int32')

                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_idx, box_idx, :]
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_idx, box_idx, :]
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5+class_idx] = 1 # one-hot encoding
        
    return y_true, y_true_boxes_xywh


