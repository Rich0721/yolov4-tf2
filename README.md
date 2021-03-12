# yolov4-tf2

This is a yolov4 program. You can train VOC or COCO datasets. The paper is [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934).  


## TODO
- [x] CSPDarknet53
- [x] Train
- [x] Predict image
- [x] Evalutate model
- [x] train coco dataset
- [ ] train voc dataset
- [ ] Predict video
- [ ] Yolov4 Tiny
- [ ] More backbone

## Requriments
I use tensorflow2 implenment.
```
pip install -r requirments.txt
```

## Train
Step 1: Modify `config.py`  
```
    EPOCHS : The number of iteration. 
    BATCH_SIZE: You can modify size, according to your GPU.

    CLASSES: You want to train classes.
    
    DATASET : Dataset path
    VOC_TEXT_FILE: Dataset train, val, test images information.
    VOC_TRAIN_FILE: Produce train text file.

    MODEL_FOLDER : Stroage weigth folder
    FILE_NAME : Storage weigth weight
```
Open the common line or IDLE.  
Step 2: `python voc_annotation.py`
You can obtain three text files(train.txt, val.txt and test.txt)  
  
Step 3: `python train_300.py`  

## Test

If you want to use my weight file, you can go to download in the link.  
https://drive.google.com/file/d/1EkN-JIqn1V9xwYWcXYsRjUfJRXpD8lPK/view?usp=sharing

You can look `predict_image.ipynb`.  

### Execute results
`python predict.py`

## Evalutate model
Modify the `evaluate.py` load weigth path, you can evaluate yourself model.   
`python evaluate.py` 

## Execute Video

`python video.py`  
You can look to detected result of video.

## Reference
[yolov4-keras](https://github.com/bubbliiiing/yolov4-keras)  
[yolo-v4-tf.keras](https://github.com/taipingeric/yolo-v4-tf.keras)  
