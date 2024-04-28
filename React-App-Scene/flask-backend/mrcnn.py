import pixellib
from pixellib.instance import instance_segmentation
import tensorflow as tf
from tensorflow.keras.models import load_model

def mrcnn_model():
    segment_image = instance_segmentation()
    segment_image.load_model('D:\\React-App-Scene\\mask_rcnn_coco.h5')
    print(dir(segment_image))

    return segment_image