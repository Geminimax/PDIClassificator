import sys
import classifier
from classifier import read_data,multiple_images_predict

image_dir_path =  sys.argv[1]
method = sys.argv[2]

if method.lower() == "haralick":
    descriptor = classifier.HARALICK
elif method.lower() == "lbp":
    descriptor = classifier.LBP
elif method.lower() == "color":
    descriptor = classifier.COLOR_HIST
else:
    descriptor = classifier.COLOR_HIST
    print("Invalid descriptor parameter, defaulting to " + descriptor);

print("Classifiying using : " + descriptor)
images,labels = read_data(image_dir_path, load_image = True)

multiple_images_predict(images,descriptor,labels)