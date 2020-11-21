import tensorflow as tf
from tensorflow.keras.layers import Concatenate, UpSampling2D
from networks.cspdarknet import conv, cspdarknet53


def yolov4_neck(x, num_classes):

    backbone = cspdarknet53(x)
    route0, route1, route2 = backbone.output
    
    route_input = route2
    x = conv(route2, 256, 1)
    x = UpSampling2D()(x)
    route1 = conv(route1, 256, 1)
    x = Concatenate()([route1, x])
    
    x = conv(x, 256, 1)
    for i in range(2):
        x = conv(x, 512, 3)
        x = conv(x, 256, 1)
    
    route1 = x
    x = conv(x, 128, 1)
    x = UpSampling2D()(x)
    route0 = conv(route0, 128, 1)
    x = Concatenate()([route0, x])

    x = conv(x, 128, 1)
    for i in range(2):
        x = conv(x, 256, 3)
        x = conv(x, 128, 1)
    
    route0 = x
    x = conv(x, 256, 3)
    conv_sbbox = conv(x, 3*(num_classes+5), 1, activation=None, batch_norm=False, use_bias=True)

    x = conv(route0, 256, 3, downsampling=True)
    x = Concatenate()([x, route1])

    x = conv(x, 256, 1)
    for i in range(2):
        x = conv(x, 512, 3)
        x = conv(x, 256, 1)
    
    route1 = x
    x = conv(x, 512, 3)
    conv_mbbox = conv(x, 3*(num_classes+5), 1, activation=None, batch_norm=False, use_bias=True)

    x = conv(route1, 256, 3, downsampling=True)
    x = Concatenate()([x, route_input])

    x = conv(x, 512, 1)
    for i in range(2):
        x = conv(x, 1024, 3)
        x = conv(x, 512, 1)
    
    x = conv(x, 1024, 3)
    conv_lbbox = conv(x, 3*(num_classes+5), 1, activation=None, batch_norm=False, use_bias=True)
    return [conv_sbbox, conv_mbbox, conv_lbbox]


def yolov4_head(yolo_neck_outputs, classes, anchors, xyscale):
    bbox0, object_prob0, class_prob0, pred_box0 = get_boxes(yolo_neck_outputs[0], anchors=anchors[0, :, :], classes=classes, grid_size=52, strides=8, xyscale=xyscale[0])
    bbox1, object_prob1, class_prob1, pred_box1 = get_boxes(yolo_neck_outputs[1], anchors=anchors[1, :, :], classes=classes, grid_size=26, strides=16, xyscale=xyscale[1])
    bbox2, object_prob2, class_prob2, pred_box2 = get_boxes(yolo_neck_outputs[2], anchors=anchors[2, :, :], classes=classes, grid_size=13, strides=32, xyscale=xyscale[2])

    x = [bbox0, object_prob0, class_prob0, pred_box0,
        bbox1, object_prob1, class_prob1, pred_box1,
        bbox2, object_prob2, class_prob2, pred_box2]
    return x


def get_boxes(pred, anchors, classes, grid_size, strides, xyscale):
    
    pred = tf.reshape(pred, (tf.shape(pred)[0], grid_size, grid_size, 3, 5+classes)) # Shape: (batch_size, grid, grid, 3, 5+classes)
    box_xy, box_wh, obj_prob, class_prob = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    obj_prob = tf.sigmoid(obj_prob)
    class_prob = tf.sigmoid(class_prob)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))# (52, 52) (52, 52)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) # (52, 52, 1, 2)
    grid = tf.cast(grid, dtype=tf.float32)

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + grid) * strides

    box_wh = tf.exp(box_wh) * anchors
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    pred_box = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return pred_box, obj_prob, class_prob, pred_box_xywh


def nms(model_ouputs, input_shape, num_class, iou_threshold=0.413, score_threshold=0.3):
    """
    Apply Non-Maximum suppression
    ref: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression
    :param model_ouputs: yolo model model_ouputs
    :param input_shape: size of input image
    :return: nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections
    """
    bs = tf.shape(model_ouputs[0])[0]
    boxes = tf.zeros((bs, 0, 4))
    confidence = tf.zeros((bs, 0, 1))
    class_probabilities = tf.zeros((bs, 0, num_class))

    for output_idx in range(0, len(model_ouputs), 4):
        output_xy = model_ouputs[output_idx]
        output_conf = model_ouputs[output_idx + 1]
        output_classes = model_ouputs[output_idx + 2]
        boxes = tf.concat([boxes, tf.reshape(output_xy, (bs, -1, 4))], axis=1)
        confidence = tf.concat([confidence, tf.reshape(output_conf, (bs, -1, 1))], axis=1)
        class_probabilities = tf.concat([class_probabilities, tf.reshape(output_classes, (bs, -1, num_class))], axis=1)

    scores = confidence * class_probabilities
    boxes = tf.expand_dims(boxes, axis=-2)
    boxes = boxes / input_shape[0]  # box normalization: relative img size
    print(f'nms iou: {iou_threshold} score: {score_threshold}')
    (nmsed_boxes,      # [bs, max_detections, 4]
     nmsed_scores,     # [bs, max_detections]
     nmsed_classes,    # [bs, max_detections]
     valid_detections  # [batch_size]
     ) = tf.image.combined_non_max_suppression(
        boxes=boxes,  # y1x1, y2x2 [0~1]
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,  # max_boxes: Maximum nmsed_boxes in a single img.
        iou_threshold=iou_threshold,  # iou_threshold: Minimum overlap that counts as a valid detection.
        score_threshold=score_threshold,  # # Minimum confidence that counts as a valid detection.
    )
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections