import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import skimage
import os
import joblib
from skimage import color, feature, exposure, io

LBP_PATH = "trained_models/knn_lbp.pkl"
HARALICK_PATH = "trained_models/knn_haralick.pkl"
COLOR_HIST_PATH = "trained_models/knn_colorHist.pkl"
LBP = "Linear Binary Pattern"
HARALICK = "Haralick Features"
COLOR_HIST = "Color Histogram"

def read_data(image_dir_path, load_image = False):
    fruits = []
    labels = []
    for image_dir in os.listdir(image_dir_path):
        full_path = os.path.join(image_dir_path,image_dir)

        if os.path.isdir(full_path):
            list_class_dir = os.listdir(full_path)

            for image in list_class_dir:
                if load_image:
                    fruits.append(io.imread(os.path.join(full_path, image)))
                else:
                    fruits.append(os.path.join(full_path, image))
                labels.append(image_dir)

    return fruits, labels

def extract_colorHist(image):
    img_f = skimage.img_as_float(image)
    hist_red = exposure.histogram(img_f[:,:,0].flatten(), nbins=8, normalize=True)[0]
    hist_green = exposure.histogram(img_f[:,:,1].flatten(), nbins=8, normalize=True)[0]
    hist_blue = exposure.histogram(img_f[:,:,2].flatten(), nbins=8, normalize=True)[0]
    hist = []
    hist.extend(hist_red)
    hist.extend(hist_green)
    hist.extend(hist_blue)

    #hist = plt.hist(img_f.flatten(),bins=np.arange(0,8),normed = True)[0]
    return hist

def extract_haralick(image):
    haralick = mh.features.haralick(image)
    haralick_mean = haralick.mean(axis=0)
    return haralick_mean

def extract_lbp(image):
    radius = 3
    n_points = radius * 8
    ##Function for lbp feature extraction
    gray_image = color.rgb2gray(image)
    lbp = feature.local_binary_pattern(gray_image, n_points, radius)
    hist = exposure.histogram(lbp, nbins=n_points, normalize = True)[0]
    return hist

def read_tests(image_dir_path):
    fruits_test = []
    labels_test = []
    for image_dir in os.listdir(image_dir_path):
        full_path = os.path.join(image_dir_path,image_dir)

        if os.path.isdir(full_path):
            list_class_dir = os.listdir(full_path)

            for image in list_class_dir:
                fruits_test.append(os.path.join(full_path, image))
                labels_test.append(image_dir)

    return fruits_test, labels_test

def multiple_images_predict(images, descriptor, labels = None):

    features_image = []
    if descriptor == LBP:
        path = LBP_PATH
        for image in images:
            features_image.append(extract_lbp(image))
    elif descriptor == HARALICK:
        path = HARALICK_PATH
        for image in images:
            features_image.append(extract_haralick(image))
    elif descriptor == COLOR_HIST:
        path = COLOR_HIST_PATH
        for image in images:
            features_image.append(extract_colorHist(image))

    model = joblib.load(os.path.abspath(path))
    predict = model.predict(features_image)
	   
    if labels is not None:
        correct = 0
        for i in range(len(predict)):
            print("Prediction: " + predict[i] + "  Expected: " + labels[i])
            if(predict[i] == labels[i]):
                correct += 1
                
        print(str(correct) + "/" + str(len(predict)))
    return predict