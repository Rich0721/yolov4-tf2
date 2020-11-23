Config = {
    "image_size":(416, 416, 3),
    'anchors':[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
    "strides": [8, 16, 32],
    'xyscale': [1.2, 1.1, 1.05],

    # image basic
    "classes": ['airwaves-mint', 'eclipse-lemon', 'eclipse-mint', 'eclipse-mint-fudge', 'extra-lemon', \
        'hallsxs-buleberry', 'hallsxs-lemon', 'meiji-blackchocolate', 'meiji-milkchocolate', 'rocher'],
    "image_folder":"../datasets/voc/JPEGImages",
    "annotation_folder": "../datasets/voc/Annotations",
    "train_text" : "../datasets/voc/train.txt",
    "val_text": "../datasets/voc/val.txt",
    "test_text": "../datasets/voc/test.txt",

    # result file
    "storage_folder":"./logs",
    "storage_weight":"./yolov4",

    # Train
    "iou_loss_thresh": 0.5,
    "batch_size": 8,
    "num_gpu": 1,
    'epochs': 100,
    # Inference
    'max_boxes': 100,
    "iou_threshold": 0.413,
    "score_threshold": 0.5
}