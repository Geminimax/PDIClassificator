import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
from sklearn import svm
from skimage import io, color, feature, data

def show(img):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()

image_dir_path = "frutas_dataset_train"
image_list = []
labels = []
count = 0
##Reads each image in subdirectories and stores it in image_list
##The directory names are used as labels for classification
print("reading images")
for image_dir in os.listdir(image_dir_path):
    full_path = os.path.join(image_dir_path,image_dir)
    
    if os.path.isdir(full_path):
        for image_file in os.listdir(full_path):
            print(os.path.join(full_path,image_file))
            image = io.imread(os.path.join(full_path,image_file))
            image_list.append(image)
            labels.append(image_dir)
#Takes about a billion years to be done

def extract_lbp(image,radius,n_points):
    ##Function for lbp feature extraction
    gray_image = color.rgb2gray(image)
    lbp = feature.local_binary_pattern(gray_image, n_points, radius)
    hist = plt.hist(lbp.ravel(),bins=np.arange(0, n_points),
                        range=(0, n_points),normed = True)[0]
    return hist

print("extracting features")
radius = 3
n_points = radius * 8

datasets = []
for image in image_list:
    datasets.append(extract_lbp(image,radius,n_points))

print("svm")
##SVM model generation
model = svm.LinearSVC(C=100.0, random_state=42,max_iter = 10000)
model.fit(datasets,labels)

##Test
test_image = io.imread("frutas_dataset_train/kiwi/kiwi_001.jpg")
test_data = extract_lbp(test_image,radius,n_points) 
predict = model.predict([test_data])
print("Prediction " + str(predict[0]))