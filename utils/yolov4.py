import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from networks.yolo import yolov4_head, yolov4_neck, nms
from utils.utils import load_weights, draw_bbox, draw_plot_func, get_detection_data
from config import Config
from loss import yolo_loss


class Yolov4(object):

    def __init__(self, weight_path=None, config=Config):

        assert config['image_size'][0] == config['image_size'][1], 'not support yet'
        assert config['image_size'][0] % config['strides'][-1] == 0, 'must be a multiple of last stride'
        self.classes_name = Config["classes"]
        self.image_size = Config['image_size']
        self.num_classes = len(self.classes_name)
        self.weight_path = weight_path
        self.anchors = np.array(Config['anchors']).reshape((3, 3, 2))
        self.xyscale = Config['xyscale']
        self.strides = Config['strides']
        self.output_sizes = [self.image_size[0] // s for s in self.strides]
        self.class_color = {name: list(np.random.random(size=3)*255) for name in self.classes_name}
        self.max_boxes = Config['max_boxes']
        self.iou_loss_thresh = Config["iou_loss_thresh"]
        self.config = Config
        self.mkdir(self.config["storage_folder"])

        K.clear_session()
        if Config['num_gpu'] >1:
            mirror_strategy = tf.distribute.MirroredStrategy()
            with mirror_strategy.scope():
                self.build_model(load_pretrained=True if self.weight_path else False)
        else:
            self.build_model(load_pretrained=True if self.weight_path else False)
    
    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def build_model(self, load_pretrained=True):
        input_layer = Input(self.image_size)
        outputs = yolov4_neck(input_layer, self.num_classes)
        self.yolo_model = Model(input_layer, outputs)

        # Build training model
        y_true = [
            Input(name='input_2', shape=(52, 52, 3, (self.num_classes + 5))), # label small boxes
            Input(name='input_3', shape=(26, 26, 3, (self.num_classes + 5))), # label medium boxes
            Input(name='input_4', shape=(13, 13, 3, (self.num_classes + 5))), # label large boxes
            Input(name='input_5', shape=(self.max_boxes, 4)) # true bboxes
        ]
        loss_list = Lambda(yolo_loss, name='yolo_loss',
                            arguments={
                                'num_classes':self.num_classes, 'iou_loss_thresh':self.iou_loss_thresh,
                                'anchors':self.anchors})([*self.yolo_model.output, *y_true])
        self.training_model = Model([self.yolo_model.input, *y_true], loss_list)

        # Build inference model
        yolov4_output = yolov4_head(outputs, self.num_classes, self.anchors, self.xyscale)
        self.inference_model = Model(input_layer, nms(yolov4_output, self.image_size, self.num_classes,
                                                    iou_threshold=self.config['iou_threshold'],
                                                    score_threshold=self.config['score_threshold']))
        
        if load_pretrained and self.weight_path and self.weight_path.endswith(".weights"):
            load_weights(self.yolo_model, self.weight_path)
        elif load_pretrained and self.weight_path and self.weight_path.endswith(".h5"):
            self.inference_model.load_weights(self.weight_path, by_name=True)
        
        self.training_model.compile(optimizer=Adam(learning_rate=1e-5), loss={'yolo_loss':lambda y_true, y_pred : y_pred})
    
    def load_model(self, path):
        self.yolo_model = models.load_model(path, compile=False)
        output = yolov4_head(self.yolo_model.output, self.num_classes, self.anchors, self.xyscale)
        self.inference_model = Model(self.yolo_model.input, nms(output, self.img_size, self.num_classes))

    def save_model(self, path):
        self.yolo_model.save(path)
    
    def preprocess_img(self, img):
        img = cv2.resize(img, self.image_size[:2])
        img = img / 255.
        return img
    
    def fit(self, train_data_gen, val_data_gen=None, initial_epoch=0, callbacks=None):
        if callbacks is None:
            callbacks = self.create_callbacks()
            self.training_model.fit_generator(train_data_gen,
                                    steps_per_epoch=len(train_data_gen),
                                    validation_data=val_data_gen,
                                    validation_steps=len(val_data_gen),
                                    epochs=self.config['epochs'],
                                    callbacks=callbacks,
                                    initial_epoch=initial_epoch)
        else:
            self.training_model.fit_generator(train_data_gen,
                                    steps_per_epoch=len(train_data_gen),
                                    validation_data=val_data_gen,
                                    validation_steps=len(val_data_gen),
                                    epochs=self.config['epochs'],
                                    callbacks=callbacks,
                                    initial_epoch=initial_epoch)
    
    def create_callbacks(self):
        model_check = ModelCheckpoint(os.path.join(self.config['storage_folder'], self.config["storage_weight"] + "-{epoch:02d}.h5"), save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        learningRateScheduler = LearningRateScheduler(self.learing_rate, verbose=1)
        return [model_check,  early_stopping,learningRateScheduler]
    
    def learing_rate(self, epoch):
        if epoch < 20:
            return 1e-6
        elif epoch < 80:
            return 1e-5
        else:
            return 1e-6

    def predict_img(self, raw_img, random_color=True, plot_img=True, figsize=(10, 10), show_text=True):
        img = self.preprocess_img(raw_img)
        imgs = np.expand_dims(img, axis=0)
        pred_output = self.inference_model.predict(imgs)
        detections = get_detection_data(img=raw_img, model_outputs=pred_output, class_names=self.classes_name)

        if plot_img:
            draw_bbox(raw_img, detections, cmap=self.class_color, random_color=random_color, figsize=figsize, show_text=show_text)
        return detections
    
    def predict(self, img_path, random_color=True, plot_img=True, figsize=(10, 10), show_text=True):
        raw_img = cv2.imread(img_path)[:, :, ::-1]
        return self.predict_img(raw_img, random_color, plot_img, figsize, show_text)

    def predcit_images(self, random_color=True, plot_img=True, figsize=(10, 10), show_text=True):
        test_text = Config["test_text"]
        test_file_names = open(test_text, 'r').read().split()

        for img in test_file_names:
            raw_img = cv2.imread(os.path.join(Config["image_folder"], img))[:, :, ::-1]
            self.predict_img(raw_img, random_color, plot_img, figsize, show_text)
    
    def evaluate(self, generator, iou_threshold=0.5):

        # gather all detection and annotations
        all_detections = [[None for i in range(generator.num_classes)] for j in range(generator.size())]
        all_annotations = [[None for i in range(generator.num_classes)] for j in range(generator.size())]

        print("Process predict bboxes!")
        for i in (range(generator.size())):
            raw_image = generator.load_image(i)
            raw_image = raw_image[:, :, ::-1]
            
            detections = self.predict_img(raw_image, random_color=False, plot_img=False)
            
            # detection(xmin, ymin, xmax, ymax, class_name, score, w, h)
            score = np.array([detection[5] for detection in detections])
            pred_labels = np.array([self.classes_name.index(detection[4]) for detection in detections])
            
            if len(detections) > 0:
                pred_boxes = np.array([[detection[0], detection[1], detection[2], detection[3], detection[5]] for detection in detections])
            else:
                pred_boxes = np.array([[]])
            
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
            
            annotations = generator.load_annotations(i)
                # copy detections to all_annotations
            for label in range(generator.num_classes):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            
        
        average_precisions = {}
        print("Compute average precisions.")
        for label in range(generator.num_classes):
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []
                
                for d in detections:

                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue
                    
                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
            
            if num_annotations == 0:
                average_precisions[self.classes_name[label]] = 0
                continue        
            
            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = compute_ap(recall, precision)
            average_precisions[self.classes_name[label]] = average_precision
        return average_precisions

def compute_overlap(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    inter_w = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    inter_h = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    inter_w = np.maximum(0, inter_w)
    inter_h = np.maximum(0, inter_h)

    union = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - inter_w * inter_h

    union = np.maximum(union, np.finfo(float).eps)
    intersect = inter_h * inter_w

    return intersect / union


def compute_ap(recall, precision):

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap