from flask import Flask
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.semantic import semantic_segmentation
from flask import send_from_directory,send_file
from unet import load_images_bw, multi_unet_model
from yolo_model import load_yolo_model
import numpy as np
from keras.utils import normalize
from matplotlib import pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import supervision as sv



################################################################

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import os

app = Flask(__name__)
CORS(app)  # Initialize CORS with your Flask app

UPLOAD_FOLDER = 'D:\\React-App-Scene\\flask-backend\\uploads'  # Folder to store uploaded files
PROCESS_FOLDER = 'D:\\React-App-Scene\\flask-backend\\processed'  # Folder to store uploaded files
app.config['PROCESS_FOLDER'] = PROCESS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


    
    


# Define the boundaries function
def add_boundaries(image, mask, color, thickness=2):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on the image
    cv2.drawContours(image, contours, -1, color, thickness)

def process_image(file_path):
    # Example function to process an image
    # For demonstration, this function simply returns the file path
    return file_path

def get_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS):
    return multi_unet_model(n_classes=14, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

class_colors = {
              # Background
    1:  (255, 0, 0),    # Wall
    2:  (0, 0, 255),    # Building
    3:  (0, 128, 255),      # Sky
    4:  (0, 0, 204),       # Floor
    5:  (0, 255, 0),        # Tree
    6:  (0, 102, 102),     # Ceiling
    7:  (192, 192, 192),    # Route
    8:  (255,255,0),        # Bed
    10: (0, 51, 25),       # grass
    13: (255, 153, 153),      # Person
    26: (153,76,0),      # House
    38: (128, 128, 128),     # bathtub  
    66: (0,153,153),       # toilet
   
}

class_labels = {
   
    1: 'Wall',
    2: 'Building',
    3: 'Sky',
    4: 'Floor',
    5: 'Tree',
    6: 'Ceiling',
    7: 'Route',
    8: 'Bed',
    10: 'grass',
    13: 'Person',
    26 : 'House',
    38:  'bathtub',  
    66:  'toilet'
}
'''
class_labels = {
        
        2: 'building',
        3: 'sky',
        5: 'tree',
        6: 'ceiling',
        7: 'road',
        10: 'grass',
        12: 'sidewalk',
        13: 'person',
        17: 'mountain',
        21: 'car',
        22: 'water',
        26: 'house',
        27: 'sea'}
class_colors = {
            2: (80, 120, 120), 
            3: (6, 230, 230),          
            5: (4, 200, 3),    
            6: (20, 120, 80),    
            7: (140, 140, 140), 
            10: (4, 250, 7), 
            12: (235, 255, 7),
            13: (150, 5, 61),              
            17: (143, 255, 140),    
            21: (0, 102, 200), 
            22: (61, 230, 250),
            26:(255, 9, 224),
            27:(9, 7, 230)              
        } 
'''

def load_unet():
    IMG_HEIGHT = 128
    IMG_WIDTH  = 128
    IMG_CHANNELS = 1
    print(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    model = get_model(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.load_weights("D:\\React-App-Scene\\weights-6-improvement-fixed-images-033-0.72.hdf5")
    
   
    #model.load_weights("D:\\React-App-Scene\\weights-14-improvement-fixed-images-049-0.68.hdf5")
    #model.load_weights("C:\\Users\\bhole\Downloads\\U_net_final\\14\\weights-14-improvement-fixed-images-003-0.82.hdf5")
    
    
    return model

def load_pixellib_resnet():
    ins = instanceSegmentation()
    ins.load_model("D:\React-App-Scene\pointrend_resnet50.pkl")
    return ins

def load_sam():
    sam = sam_model_registry["vit_l"](checkpoint="D:\\React-App-Scene\\sam_vit_l_0b3195.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator

    
def load_deeplab():
    segment_video = semantic_segmentation()
    segment_video.load_ade20k_model("D:\\React-App-Scene\\deeplabv3_xception65_ade20k.h5")
    return segment_video

@app.route('/upload/pixellib', methods=['POST'])
def process_pixellib_resnet50():
    ins = load_pixellib_resnet()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        #processed_file_path = process_image(file_path)
        nw_file = file.filename.split('.')[0] + '_seg'+'.png'
        output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)
        ins.segmentImage(file_path, output_image_name = output_image_name, show_bboxes = True)
        processed_filename = output_image_name.split('processed\\')[1]
        processed_file_destination = os.path.join(app.config['PROCESS_FOLDER'], processed_filename)
        # Return the URL of the processed image
        processed_image_url = os.path.join(app.config['PROCESS_FOLDER'], processed_file_destination)
        with open(processed_image_url, 'rb') as f:
            image_data = f.read()
            return image_data


segment_video = load_deeplab()
@app.route('/upload/deeplab', methods=['POST'])
def process_pixellib_deeplab():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        #processed_file_path = process_image(file_path)
        nw_file = file.filename.split('.')[0] + '_seg'+'.png'
        output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)
        segment_video.segmentAsAde20k(file_path, output_image_name = output_image_name)
        processed_filename = output_image_name.split('processed\\')[1]
        processed_file_destination = os.path.join(app.config['PROCESS_FOLDER'], processed_filename)
        # Return the URL of the processed image
        processed_image_url = os.path.join(app.config['PROCESS_FOLDER'], processed_file_destination)
        with open(processed_image_url, 'rb') as f:
            image_data = f.read()
            return image_data



@app.route('/upload/unet', methods=['POST'])
def unet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
    image = load_images_bw(file_path)
    train_images = np.array(image[:,:,0])
    train_images = np.expand_dims(train_images, axis=2)
    train_images = normalize(train_images,axis=1)

    model = load_unet()
    test_img_input=np.expand_dims(train_images, 0)

    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    print(np.unique(predicted_img))
    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=1000)
    #plt.subplot(211)
    #plt.title('Testing Image')
    #plt.imshow(image)

    #plt.subplot(212)
    #plt.title('Prediction on test image')
    #plt.imshow(predicted_img,cmap='gray')
    
    
    # Apply colors to the mask based on class labels
    output = np.repeat(predicted_img[:, :, np.newaxis], 3, axis=2)
    labels = list(class_colors.values())
    for class_label, color in class_colors.items():
        output[predicted_img == class_label] = color  
     
    
    plt.imshow(image)
    plt.imshow(output,alpha=0.60)
    plt.axis('off')

    nw_file = file.filename.split('.')[0] + '_seg'+'.png'
    output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)
    
    # Save the image shown in subplot (222)
    plt.savefig(output_image_name,bbox_inches='tight', pad_inches=0,dpi='figure')
    #processed_file_destination = os.path.join(app.config['PROCESS_FOLDER'], processed_filename)
    with open(output_image_name, 'rb') as f:
            image_data = f.read()
            return image_data
 
    return jsonify({'message': 'File processed for API 4'})
        
@app.route('/upload/live',methods=['POST'])
def live():
    return jsonify({'message': 'File processed for API 4'})

@app.route('/upload/mrcnn', methods=['POST'])
def mrcnn():
    #segment_image = load_mrcnn()
    # Process file upload for API 3
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
    nw_file = file.filename.split('.')[0] + '_seg'+'.png'
    output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)
    
    #mrcnn_model.segmentImage(file_path, output_image_name = output_image_name,show_bboxes = True)
    processed_filename = output_image_name.split('processed\\')[1]
    processed_file_destination = os.path.join(app.config['PROCESS_FOLDER'], processed_filename)
    # Return the URL of the processed image
    processed_image_url = os.path.join(app.config['PROCESS_FOLDER'], processed_file_destination)
    with open(processed_image_url, 'rb') as f:
        image_data = f.read()
        return image_data

    
    
    return jsonify({'message': 'File processed for API 3'})

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_file(filename, mimetype='image/png')

@app.route('/upload/yolo', methods=['POST'])
def YOLO():
    # Process file upload for API 4
    yolo_model = load_yolo_model()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        #processed_file_path = process_image(file_path)
        nw_file = file.filename.split('.')[0] + '_seg'+'.png'
        output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)

        results = yolo_model.predict(file_path, conf=0.4)  #Adjust conf threshold
        new_result_array = results[0].plot()
        plt.imshow(new_result_array)

        # Save the image shown in subplot (222)
        plt.savefig(output_image_name)

        with open(output_image_name, 'rb') as f:
            image_data = f.read()
            return image_data

    return jsonify({'message': 'File processed for API 4'})

@app.route('/upload/sam', methods=['POST'])
def sam():
    # Process file upload for API 4
    mask_generator = load_sam()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

    nw_file = file.filename.split('.')[0] + '_seg'+'.png'
    output_image_name = os.path.join(app.config['PROCESS_FOLDER'], nw_file)

    
    image_bgr = cv2.imread(file_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    masks = mask_generator.generate(image)
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(masks)
    annotated_image = mask_annotator.annotate(image_bgr, detections)
    cv2.imwrite(output_image_name,annotated_image)
    with open(output_image_name, 'rb') as f:
            image_data = f.read()
            return image_data

    return jsonify({'message': 'File processed for API 4'})

@app.route('/upload/video', methods=['POST'])
def video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        #processed_file_path = process_image(file_path)
        nw_file = u"C:\\Users\\bhole\\Downloads\\New folder (2)\\3066445-uhd_4096_2160_24fps.mp4"
        #"C:\Users\bhole\Downloads\New folder (2)\3066445-uhd_4096_2160_24fps.mp4"
        output_image_name =  "C:\\Users\\bhole\\Downloads\\New folder (2)\\op.mp4"
        #segment_video.segmentAsAde20k(file_path, output_image_name = output_image_name)
        segment_video.process_video_ade20k(nw_file,  frames_per_second= 15, output_video_name=output_image_name)
        processed_filename = output_image_name
        processed_file_destination = os.path.join(app.config['PROCESS_FOLDER'], processed_filename)
        # Return the URL of the processed image
        processed_image_url = os.path.join(app.config['PROCESS_FOLDER'], processed_file_destination)
        with open(processed_image_url, 'rb') as f:
            image_data = f.read()
            return image_data

if __name__ == '__main__':
    app.run(debug=True)