import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling2D, Concatenate, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from networks.model import CSPdarknet53, DarknetConv2D, DarknetConv2D_BN_Leaky, compose

def make_last_layers(inputs, filters):

    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(inputs)
    x = DarknetConv2D_BN_Leaky(filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(filters, (1, 1))(x)
    return x


def yolo_body(inputs, num_anchors, num_classes):

    feat1, feat2, feat3 = CSPdarknet53(inputs)
    
    # SPP net
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    maxpool1 = MaxPooling2D((13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D((9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D((5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1, 1))(P5)

    P5_upsample = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(P5)
    
    P4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_last_layers(P4, 256)

    P4_upsample = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(P4)

    P3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_last_layers(P3, 128)

    # yolo_output3 shape(b, 52, 52, 3, num_classes + 5)
    P3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P3)
    P3_output = DarknetConv2D(num_anchors * (num_classes+5), (1, 1))(P3_output)
    
    P3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(P3_downsample)
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_last_layers(P4, 256)

    # yolo_output2 shape(b, 26, 26, 3, num_classes + 5)
    P4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)
    
    P4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(P4_downsample)
    
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_last_layers(P5, 512)

    # yolo_output1 shape(b, 13, 13, 3, num_classes + 5)
    P5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)
    
    return Model(inputs, [P5_output, P4_output, P3_output])

    
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)

    anchors_tensor = K.reshape(K.constant(anchors), (1, 1, 1, num_anchors, 2))

    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), (-1, 1, 1, 1)), (1, grid_shape[1], 1, 1))
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), (1, -1, 1 ,1)), (grid_shape[0], 1, 1, 1))
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (b, 13, 13, num_anchors, num_classes+5)
    feats = K.reshape(feats, (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5))

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape - new_shape) /2. /input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxs = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxs[..., 0:1],
        box_maxs[..., 1:2]
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, (-1, 4))
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, (-1, num_classes))
    return boxes, box_scores


# image prediction
def yolo_eval(yolo_outputs, anchors, num_classes, image_shape, max_boxes=20, score_threshold=0.5, iou_threshold=0.5):

    num_layers = len(yolo_outputs)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchors_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):

        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        
        # nms
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_