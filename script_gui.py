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