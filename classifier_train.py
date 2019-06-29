import numpy as np
import matplotlib.pyplot as plt
import skimage
import os
import random
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import io, color, feature, exposure

def read_data(image_dir_path):
    fruits = []
    labels = []
    for image_dir in os.listdir(image_dir_path):
        full_path = os.path.join(image_dir_path,image_dir)

        if os.path.isdir(full_path):
            list_class_dir = os.listdir(full_path)

            for image in list_class_dir:
                fruits.append(os.path.join(full_path, image))
                labels.append(image_dir)

    return fruits, labels


image_dir_path = "frutas_dataset_train"
fruits, labels = read_data(image_dir_path)
fruits_train, fruits_test, labels_train, labels_test = train_test_split(fruits, labels, random_state=2010, test_size=0.5)


def create_heatmap(true, predict, labels):
	cfmx = confusion_matrix(true, predict, labels)
	row_sums = cfmx.sum(axis=1)
	data = cfmx / row_sums[:, np.newaxis]

	print(data)

	strategy_path = []

	# # Computing strategies
	for d in labels:
	 	strategy_path.append(d)

	#  Finishing Touches
	fig,ax=plt.subplots()

	ax.set_xticks(np.arange(0,len(strategy_path)))
	ax.set_yticks(np.arange(0,len(strategy_path)))
	 
	plt.imshow(data, cmap=plt.cm.gnuplot, interpolation='nearest')
	plt.colorbar()

	# Here we put the x-axis tick labels
	# on the top of the plot.  The y-axis
	# command is redundant, but inocuous.
	ax.xaxis.tick_top()
	ax.yaxis.tick_left()
	# similar syntax as previous examples
	ax.set_xticklabels(strategy_path,minor=False,fontsize=12,rotation=90)
	ax.set_yticklabels(strategy_path,minor=False,fontsize=12)

	plt.savefig('heatmap_CH.pdf', bbox_inches='tight')
	plt.show()

def train_model(feature_set, test_set, labels_train, labels_test, model_name, labels):
    ##SVM model generation
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(feature_set,labels_train)
    
    joblib.dump(model, 'trained_models/' + model_name)
    
    predict = model.predict(test_set)
    
    single_labels = list(set(labels))
    create_heatmap(labels_test, predict, single_labels)


def generate_predictor(train_set, test_set, labels_train, labels_test, feature_extraction_method, model_name):
    feature_set = []
    for image in train_set:
        feature_set.append(feature_extraction_method(io.imread(image)))
    
    test_feature_set = []
    for image in test_set:
        test_feature_set.append(feature_extraction_method(io.imread(image)))
    
    train_model(feature_set, test_feature_set, labels_train, labels_test, model_name, labels)

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

## Function for hog feature extraction
def extract_hog(image):
    # feat is the feacture vector
    feat, hog_image = feature.hog(image, visualize=True, feature_vector=True)
    # show(hog_image)
    # print(feat)
    return feat

def extract_lbp(image):
    radius = 3
    n_points = radius * 8
    ##Function for lbp feature extraction
    gray_image = color.rgb2gray(image)
    lbp = feature.local_binary_pattern(gray_image, n_points, radius)
    hist = plt.hist(lbp.ravel(),bins=np.arange(0, n_points + 3),
                        range=(0, n_points + 2),density = True)[0]
    return hist

generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_colorHist, 'knn_colorHist.pkl')
#generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_hog, 'knn_hog.pkl')
#generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_lbp, 'knn_lbp.pkl')