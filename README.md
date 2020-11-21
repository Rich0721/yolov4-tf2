# yolov4-tf2

I reference [yolo-v4-tf.keras](https://github.com/taipingeric/yolo-v4-tf.keras) modified some input and read data.

## TODO
- [x] yolov4
- [x] train self datasets
- [x] Inference image
- [ ] More backbone
- [ ] Data Augmentation
- [ ] Inference Video


## requirments
```pip install -r requirments.txt```

## train self datasets

If you want to train self, you need image files and annotation files.
Modified config.py
```
"classes": you want to train classes name.
"image_folder" : Path of image folder.
"annotation_folder": Path of annotation folder (xml files)
"train_text" : Path of train text file.(every image have a annotation that is corresponding file name)
"val_text": same of train_text.
```
```
python train.py
```

## Inference image
```
python inference
```
