# Visual_Assignment_2
## This is the repository for my second assignment in my Visual Analytics portfolio.

### Description of project
This project consists of two scripts, both performing classification-tasks on color-images; the first one uses Logistic Regression and the second uses Neural Networks. 

### Method
As mentioned above this project contains two scripts - this means that I have essentially used two different methods in order to compare results and see which of the methods performed better on classifyng color-images. 
1) Logistic regression:
- The "logistic_regression.py" script uses cv2 in order to normalize the images, turning them into greyscale, and then uses Scikit-Learn's LogisticRegression model. After defining the classifier the script uses the predict function in order to predict labels based on the test data. Lastly it produces a classification report using Scikit-Learn's function and prints the report to a .txt file in the out-folder.
2) Convolutional Neural Network
- The "nn_classifier.py" script uses the same method in order to normalize the images as the logistic regression script does (CV2) - further it uses Scikit-learn's label binarizer. After normalizing the data and binarizing the labels the script defines a neural network and fits it on the training data. After this it makes predicions on the testdata, creating once again a classification report to see how well the clasiffier performs on the data. The report is saved to the "out" folder. 

### Usage 
In order to reproduce the results I have gotten (and which can be found in the "out" folder), a few steps has to be followed - this applies to both of the individual scripts:
1) Install the relevant packages - the list of the prerequisites for each script can be found in the requirements.txt
2) Make sure to place the script in the "src" folder. The data used in my code (cifar10) is fetched from tensorflow using the load_data() function, so nothing has to go into the in-folder. Had you wanted to use the script on a similar dataset, you would have to change up the loading of the data part of the script and place the data in the in folder. For these script, however, the "in" folder becomes redundant. 
3) Run the script from the terminal and remember to pass the required arguments (-ds (name of the dataset))

This will give you the same results as I have gotten in your "out" folder".

### Results 
The aim of this project has been to compare the results of classification tasks using respectively Logistic Regression and Convolutional Neural Networks. The results for both of the classification scripts, however, are not the best - the CNN wins on accuracy with a maximum precision score of 57% on one of the classes (airplane), while the Logistic Regression peaks on a 38% precision score on the label "truck". These results then, I can conclude, are quit poor - however it does paint i picture of the neural network as a better performing classification tool. Both of the full classification reports can be found in the "out" folder. 
