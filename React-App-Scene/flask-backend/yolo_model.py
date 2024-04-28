from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image


def load_yolo_model():
    my_new_model = YOLO('D:\\React-App-Scene\\YOLO_weights_2_best.pt')
    #my_new_model = YOLO(model='D:\React-App-Scene\YOLO_Weights_best.pt')
    
    return my_new_model