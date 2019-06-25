import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import random
from sklearn import svm
from skimage import io, color, feature, exposure

def show(img):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()

image_dir_path = "frutas_dataset_train"
#image_list = []
labels = []
image_limit_per_folder = 5

##Reads each image in subdirectories and stores it in image_list
##The directory names are used as labels for classification
#current_image_count = 0
#for image_dir in os.listdir(image_dir_path):
#    full_path = os.path.join(image_dir_path,image_dir)
#    if os.path.isdir(full_path):   
#        for image_file in os.listdir(full_path):
#            image = io.imread(os.path.join(full_path,image_file))
#            image_list.append(image)
#            labels.append(image_dir)
#            current_image_count += 1
#            if(current_image_count >= image_limit_per_folder):
#                current_image_count = 0
#                break

train_set = []
test_set = []
for image_dir in os.listdir(image_dir_path):
    full_path = os.path.join(image_dir_path,image_dir)
    
    if os.path.isdir(full_path):
        list_imgs = os.listdir(full_path)
        num_imgs = len(list_imgs)
        #Divides each fruit folder in two: training and test
        image_limit_per_folder = num_imgs/2
        current_image_count = 0
        
        #Assemble training set taking one random image per iteration
        while(current_image_count < image_limit_per_folder):
            random.seed(2010)
            img_id = random.randint(0, num_imgs-1)
            print(list_imgs[img_id])
            image = io.imread(os.path.join(full_path, list_imgs.pop(img_id)))
            train_set.append(image)
            labels.append(image_dir)
            current_image_count += 1
            num_imgs-=1
        
        #Assemble testset with what is left of this directory
        for image_file in list_imgs:
            image = io.imread(os.path.join(full_path,image_file))
            test_set.append(image)

#print(labels)
#Takes about a billion years to be done

#for image in images:
#    show(image)


def apply_model(feature_set, test_set, labels):
    ##SVM model generation
    model = svm.LinearSVC(C=100)
    model.fit(feature_set,labels)
    
    ##Test
#    test_image_count = 20
#    test_images = []
#    expected = []
    predict = model.predict(test_set)
    
    correct = 0
    for i in range(len(predict)):
        print("Prediction: " + predict[i] + "  Expected: " + labels[i])
        if(predict[i] == labels[i]):
            correct += 1
            
    print(str(correct) + "/" + str(len(predict)))

def generate_predictor(train_set, test_set,labels, feature_extraction_method):
    feature_set = []
    for image in train_set:
        feature_set.append(feature_extraction_method(image))
    
    test_feature_set = []
    for image in test_set:
        test_feature_set.append(feature_extraction_method(image))

    apply_model(feature_set, test_feature_set, labels)

def extract_lbp(image):
    radius = 3
    n_points = radius * 8
    ##Function for lbp feature extraction
    gray_image = color.rgb2gray(image)
    lbp = feature.local_binary_pattern(gray_image, n_points, radius)
    hist = plt.hist(lbp.ravel(),bins=np.arange(0, n_points + 3),
                        range=(0, n_points + 2),density = True)[0]
    return hist

generate_predictor(train_set, test_set, labels, extract_lbp)

## Function for hog feature extraction
def extract_hog(image):
    # feat is the feacture vector
    # hog_image is the result of the tranformation,
    # We can just throw it away, i think
    feat, hog_image = feature.hog(image, visualize=True, feature_vector=True)
    # show(hog_image)
    # print(feat)
    return feat

#generate_predictor(train_set, test_set, labels, extract_hog)
#using_hog(train_set, test_set, labels)

def extract_colorHist(image):
    img_f = skimage.img_as_float(image)
    hist_red = exposure.histogram(img_f[:,:,0].flatten(), nbins=range(8), normalize=True)[0]
    hist_green = exposure.histogram(img_f[:,:,1].flatten(), nbins=range(8), normalize=True)[0]
    hist_blue = exposure.histogram(img_f[:,:,2].flatten(), nbins=range(8), normalize=True)[0]
    #hist = hist_red + hist_green + hist_blue
    hist = []
    hist.extend(hist_red)
    hist.extend(hist_green)
    hist.extend(hist_blue)
    #hist = plt.hist(img_f.flatten(),bins=np.arange(0,8),normed = True)[0]
    return hist

#generate_predictor(train_set, test_set, labels, extract_colorHist)
#using_colorHist(train_set, test_set, labels)