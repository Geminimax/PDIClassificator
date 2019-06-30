import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import io
from classifier import read_data,extract_haralick,extract_lbp,extract_colorHist

image_dir_path = "frutas_dataset_train"
print("READING DATA")
fruits, labels = read_data(image_dir_path)
print("DONE")
print("SPLITTING TEST")
fruits_train, fruits_test, labels_train, labels_test = train_test_split(fruits, labels, random_state=2010, test_size=0.5)
print("DONE")

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

	plt.savefig('heatmap_HOG.pdf', bbox_inches='tight')
	plt.show()

def train_model(feature_set, test_set, labels_train, labels_test, model_name, labels):
    ##SVM model generation
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(feature_set,labels_train)
    
    print("DONE")
    joblib.dump(model, 'trained_models/' + model_name)
    
    print("TESTING MODEL")
    predict = model.predict(test_set)
    print("DONE")

    print("GENERATING HEATMAP")
    single_labels = list(set(labels))
    create_heatmap(labels_test, predict, single_labels)
    print("DONE")


def generate_predictor(train_set, test_set, labels_train, labels_test, feature_extraction_method, model_name):
    feature_set = []

    print("EXCTRACTING TRAIN FEATURES")
    i=0
    for image in train_set:
        print(str(i) + " train feat " + image)
        feature_set.append(feature_extraction_method(io.imread(image)))
        i+=1

    print("DONE")
    print("EXTRACTING TEST FEATURES")
    i=0
    test_feature_set = []
        
    for image in test_set:
        print(str(i) + " test feat " + image)
        test_feature_set.append(feature_extraction_method(io.imread(image)))
        i+=1

    print("DONE")
    
    print("TRAINING MODEL")
    train_model(feature_set, test_feature_set, labels_train, labels_test, model_name, labels)

generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_colorHist, 'knn_colorHist.pkl')
generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_haralick, 'knn_haralick.pkl')
generate_predictor(fruits_train, fruits_test, labels_train, labels_test, extract_lbp, 'knn_lbp.pkl')