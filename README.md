# yolov4-tf2

This is a yolov4 program. But I didn't train VOC or COCO datasets. The paper is `[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)`.  


## TODO
- [x] CSPDarknet53
- [x] Train
- [x] Predict image
- [x] Evalutate model
- [ ] Predict video
- [ ] Yolov4 Tiny
- [ ] More backbone

## Training 
You need to modify `config.py`    
`CLASSES`: Youself classes.  
`TRAIN_TEXT`: Train datasets  
`VAL_TEXT`: Validation datasets, it is used to evalutate model.  
`DATASETS` and `VAL_DATASET`: You can use `voc_annotation` produce `train text`. The format is (image_path xmin, ymin, xmax, ymax, classes_number.  

## Predict
```
python predcit.py
```


## Reference
[yolov4-keras](https://github.com/bubbliiiing/yolov4-keras)  
[yolo-v4-tf.keras](https://github.com/taipingeric/yolo-v4-tf.keras)  
