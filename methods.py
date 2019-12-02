import os
from paths import * # Our paths in paths.py

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from IPython.display import clear_output
from sklearn import preprocessing

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from imageai.Detection import ObjectDetection

# Taken from os.path.join(DATASET_DIR, 'camera/camera_intrinsic.txt')
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

def find_sensitivity(ground_truth,detections):
    TP,FP,FN = 0,0,0
    
    xs,ys,original_string_list = get_img_coords_with_original_string(ground_truth)    
    numDetections = len(detections)
    numTruth = len(xs)
    
    dontCheckDetections = []
    TP_points = []
    
    FP = len(detections) # Assuming all detected points are False Positives, until proven True Positives

    for key,x in enumerate(xs):
        x = xs[key]
        y = ys[key]
        
        for count,det in enumerate(detections):           
            
            det = det["box_points"]
                
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            
            # (ix,iy) are its top-left coordinates, and (ax,ay) its bottom-right coordinates.
            # ix = x1, ax = x2
            # iy = y1, ay = y2
            
            if( x1 <= x and x <= x2 and y1 <= y and y <= y2 and det not in dontCheckDetections):
                #print(f"({x},{y}) is within {det}")
                mean_x = (x1+x2)/2
                mean_y = (y1+y2)/2
                TP_points.append({"original_string":original_string_list[key],"box":det,
                                 "x1":x1, "y1":y1, "x2":x2, "y2":y2})
                TP += 1                
                FP -= 1
                
                dontCheckDetections.append(det) # Don't check this bounding box again for other points
                
                break
            
            # If no detection is observed for this true point
            if(count+1>=numDetections):
                FN += 1
                
    return TP,FP,FN,TP_points

def predict_using_detector(detector,image,extract_detected_objects=False,detection_probability=50):
    returned_results = detector.detectCustomObjectsFromImage(
            custom_objects=detector.CustomObjects(car=True),
            input_image=image, 
            input_type="array",
            output_type="array",
            minimum_percentage_probability=detection_probability,
            extract_detected_objects=extract_detected_objects)
    return returned_results


def format_inputs_appended_with_image(ImageIdSubId,extra_inputs,detector_name='test'):
    ImageIdSubIdList = ImageIdSubId.split("-")
    ImageId = ImageIdSubIdList[0]
    SubId = ImageIdSubIdList[1]
    imagePath = os.path.join(DETECTIONS_DIR, detector_name+"/"+ImageId+".jpg-objects/car-"+SubId+".jpg")
    
    inputs_extra = np.array(extra_inputs, dtype="float32").reshape(4,-1)    
    min_max_scaler = preprocessing.MinMaxScaler()
    inputs_extra_scaled = min_max_scaler.fit_transform(inputs_extra)
    inputs_extra_scaled = inputs_extra_scaled.reshape(1,-1)
    
    temp_array = img_to_array(open_image(imagePath,square_size=(48,48)))/255.0
    inputs_extra_weighted = np.full((48*12,4),np.array(inputs_extra_scaled))
    temp_array = np.append(temp_array,inputs_extra_weighted).reshape(48,48,-1)
    return temp_array

def detector_imageai(detectorName='yolo'):
    
    if detectorName == 'resnet':
        
        # From https://www.kaggle.com/hypocrites/simple-eda-with-imageai-object-detection
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(IMAGEAIWEIGHTS, 'resnet50_coco_best_v2.0.1.h5'))
        detector.loadModel()
        
    elif detectorName == 'yolo':

        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(IMAGEAIWEIGHTS, 'yolo.h5'))
        detector.loadModel()
    
    return detector

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    
    # From https://www.kaggle.com/hocop1/centernet-baseline
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def get_img_coords(s):
    
    # From https://www.kaggle.com/hocop1/centernet-baseline
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

def get_img_coords_with_original_string(s):
    
    coords = str2coords(s)
    original_string = [c for c in coords]
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys, original_string


def hide_masked_area(img, mask, th=32):
    
    # From https://www.kaggle.com/hypocrites/simple-eda-with-imageai-object-detection
    
    mask[mask >= th] = 255
    mask[mask < th] = 0

    img_acc = img.astype(np.int32) + mask
    img[img_acc > 255] = 255
    return img

def return_img(imageId,datasetType='train'):
    
    # From https://www.kaggle.com/hypocrites/simple-eda-with-imageai-object-detection
    
    img_path = os.path.join(DATASET_DIR, datasetType+'_images', '{}.{}'.format(imageId, 'jpg'))
    img = cv2.imread(img_path,)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Yolo can be affected by part of car on which camera is installed
    poly_to_hide_car = np.array([[1100,2400],[3000,2480], [3384,2640],[3384,2710], [800,2710]])
    img = cv2.fillConvexPoly(img, poly_to_hide_car, [0,0,0])
    
    mask_path = os.path.join(DATASET_DIR, datasetType+'_masks', '{}.{}'.format(imageId, 'jpg'))
    mask = cv2.imread(mask_path)
    
    if mask is None:
        return img    
    return hide_masked_area(img, mask)

def open_image_preprocessed(img_path,hide_own_bonnet=True,mask_path=None):
        
    img = cv2.imread(img_path,)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if hide_own_bonnet:
        # Yolo can be affected by part of car on which camera is installed
        poly_to_hide_car = np.array([[1100,2400],[3000,2480], [3384,2640],[3384,2710], [800,2710]])
        img = cv2.fillConvexPoly(img, poly_to_hide_car, [0,0,0])
    
    if mask_path is not None:
        mask = cv2.imread(mask_path)
        if mask is not None:
            return hide_masked_area(img, mask)   
    return img

def open_image(imagePath,square_size=(48,48)):
    img = cv2.imread(imagePath,)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dimensions = img.shape
    heightImage, widthImage, channels = dimensions
    #square_size = max(heightImage,widthImage), max(heightImage,widthImage)
    img = resizeAndPad(img, square_size, padColor=0)
    
    return img
    
def visualize_image(imagePath,width=10,title=""):
    img = open_image(imagePath)
    dimensions = img.shape
    heightImage, widthImage, channels = dimensions
    # Calculate Aspect Ratio Maintained Height
    width = min(widthImage,width)
    height = width * (heightImage/widthImage)
    size = (width, height)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img/255)
    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
    
def resizeAndPad(img, size, padColor=0):

    # From https://stackoverflow.com/a/44724368/3743430
    
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def show_image(imageId=None, img=None, width=18, addMask=True, markCars=False):
    
    if imageId is not None:
        # Read Image Using OpenCV lib
        img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(imageId, 'jpg'))
        img = cv2.imread(img_path, 1)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    dimensions = img.shape
    heightImage, widthImage, channels = dimensions
    # Optionally Show Dimension
    # print(dimensions)
    
    # Calculate Aspect Ratio Maintained Height
    width = min(widthImage,width)
    height = width * (heightImage/widthImage)
    size = (width, height)
    
    # Draw Image in SubPlot
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img/255)
    ax.set_title(imageId)
    
    if (addMask):
        
        # Get corresponding mask
        mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(imageId, 'jpg'))
        mask = cv2.imread(mask_path, 0)

        patches = []
        _,contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor='black', facecolor='black', fill=True)
            patches.append(poly_patch)
        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)

        ax.add_collection(p)
    
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    
    if(markCars is not False):
        plt.scatter(*get_img_coords(markCars), color='red', s=100);
    
    plt.show()
    