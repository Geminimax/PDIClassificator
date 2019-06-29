import classifier_train
import joblib

def single_image_predict(image, descriptor, path):

	features_image = []
	if descriptor == "LBP":
		features_image.append(classifier_train.extract_lbp(image))
	elif descriptor == "HOG":
		features_image.append(classifier_train.extract_hog(image))
	elif descriptor == "ColorHist":
		features_image.append(classifier_train.extract_colorHist(image))

	model = joblib.load(path)
	predict = model.predict(features_image)
	return predict

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

def multiple_images_predict(images, descriptor, path):

	features_images = []
	if descriptor == "LBP":
		for image in images:
			features_image.append(classifier_train.extract_lbp(image))
	elif descriptor == "HOG":
		for image in images:
			features_image.append(classifier_train.extract_hog(image))
	elif descriptor == "ColorHist":
		for image in images:
			features_image.append(classifier_train.extract_colorHist(image))

	model = joblib.load(path)
	predict = model.predict(features_image)
	   
	correct = 0
    for i in range(len(predict)):
        print("Prediction: " + predict[i] + "  Expected: " + labels_test[i])
        if(predict[i] == labels_test[i]):
            correct += 1
            
    print(str(correct) + "/" + str(len(predict)))