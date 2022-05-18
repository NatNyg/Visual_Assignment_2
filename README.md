# Visual_Assignment_2
## This is the repository for my second assignment in my Visual Analytics portfolio.

### Description of project
This project consists of two scripts, both performing classification-tasks on images; the first one uses Logistic Regression and the second uses Neural Networks. The script is set up so that it can take the input of either MNIST784 (black/white images) or CIFAR10 (color images)

### Method
As mentioned above this project contains two scripts - this means that I have essentially used two different methods in order to compare results and see which of the methods performed better on classifying images. 
1) Logistic regression:
- The "logistic_regression.py" script uses cv2 in order to normalize the images, turning them into greyscale, and then uses Scikit-Learn's LogisticRegression model. After defining the classifier the script uses the predict function to predict labels based on the test data. Lastly it produces a classification report using Scikit-Learn's function and prints the report to a .txt file in the out-folder.
2) Convolutional Neural Network
- The "nn_classifier.py" script uses the same method in order to normalize the images as the logistic regression script does (CV2) - further it uses Scikit-learn's label binarizer. After normalizing the data and binarizing the labels the script defines a neural network and fits it on the training data. After this it makes predictions on the testdata, creating once again a classification report to see how well the classifier performs on the data. The report is saved to the "out" folder. 

### Usage 
In order to reproduce the results I have gotten (and which can be found in the "out" folder), a few steps has to be followed - this applies to both of the individual scripts:
1) Install the relevant packages - the list of the prerequisites for the scripts can be found in the requirements.txt
2) Make sure to place the script in the "src" folder. Further, there is a "utils" folder in the src folder, which contains an additional script "neuralnetwork.py", which is the neural network I use in my main script "nn_classification.py" - due to this, it is important that the utils folder containing the neuralnetwork.py is also placed in the src folder, as I have done in my repository. The data used in my code (cifar10) is fetched from tensorflow using the load_data() and mnist is fetched using the fetch_openml function, so nothing has to go into the in-folder. Had you wanted to use the script on a similar dataset, you would have to change up the loading of the data part of the script and place the data in the in folder. For these scripts, however, the "in" folder becomes redundant. 
3) Run the script from the terminal and remember to pass the required argument (-ds {name of the dataset})

This should give you approximately the same results as I have gotten in the "out" folder".

### Results 
The aim of this project has been to compare the results of classification tasks using respectively Logistic Regression and Convolutional Neural Networks. 
CIFAR10: For CIFAR 10 the results for both classification scripts, however, are not the best - for my results the CNN wins on accuracy with a maximum precision score of 57% on the airplane label, while the Logistic Regression peaks on a 40% precision score on the label "truck". These results then, I can conclude, are not the best - however it does paint a picture of the neural network as a better performing classification tool. 
MNIST784: For the mnist data the results are quite different, and way better. These results have a precision around about 90% which is pretty good! This shows that the mnist data is easier to make classification on - which makes sense, since it is all black and white containing white numbers on black backgrounds. Further, the fact that it is already black and white means that it doesn't have to be normalized in terms of this like CIFAR 10 (a dataset containing color images) does, in order to perform classification using these methods does. This results in the CIFAR 10 dataset losing one of the vital features for image classification; colours. 
All full classification reports I have generated can be found in the "out" folder for further inspection.
