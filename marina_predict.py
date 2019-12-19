# import the packages

from keras.models import load_model
import argparse
# load the label binarizer
import pickle
# for annotation and display
import cv2

# construct the argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True, help="path to trained keras model")
ap.add_argument("-l", "--label-bin", required=True, help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28, help="target spatial dimention width")
ap.add_argument("-e", "--height", type=int, default=28, help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1, help="whther or not we should flatten the image")

args = vars(ap.parse_args())

# load the imput image and resize it to target spatial dimensions

image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))

# scale the pizel values to [0,1]
image = image.astype("float")/255.0

# check to see if we should flatten the image and add a batch dimension - this is for a standard fully connected network

if args["flatten"] > 0:
    image = image.flatten()
    image = image.reshape((1,image.shape[0]))

# otherwise-working with CNN- don't flatten
else: 
    image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2])) 

# load the model and label binarizer
print("[INFO] Loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# make a prediction on the image
preds = model.predict(image)

# findthe class label index with larges corresponding probability
# finds the index of the max value
ind = preds.argmax(axis=1)[0]
# extracts string label from the label binarizer
label = lb.classes_[ind]

# draw the class label + prob on output image
text = "{}: {:.2f}%".format(label, preds[0][ind]*100)
cv2.putText(output, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

# show the output image
#cv2.imshow("Image", output)

cv2.imwrite('./output/{}_simple_nn_pred_result.jpg'.format(label), output)
#cv2.waitKey(0)




	









