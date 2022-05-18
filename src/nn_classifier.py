"""
Firstly, let's import the libraries we'll be using for the script!
"""
# path tools
import sys
import os
import argparse 

# image processing 
import cv2 

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10
from utils.neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn import metrics



def nn_classifier(data_set):
    """
This function takes the input dataset (cifar10 or mnist784) from the terminal, prepares the data by splitting, scaling and binarizing labels  and uses a neural network to perform classification by first fitting the network on the train data, then predicting on the test data. Finally, the function saves a classification report on the results of the prediction vs true labels to the "out" folder. 
    """
    if data_set == "CIFAR10":
        
        labels = ["airplane",
                  "automobile",
                  "bird",
                  "cat",
                  "deer",
                  "dog",
                  "frog",
                  "horse",
                  "ship",
                  "truck"]
        
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   
        X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
        X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    
        X_train_scaled = X_train_grey/255
        X_test_scaled = X_test_grey/255
    
        nsamples, nx, ny = X_train_scaled.shape
        X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
        nsamples, nx, ny = X_test_scaled.shape
        X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
        
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)
        print("[INFO] training network...")
        input_shape = X_train_dataset.shape[1]
        nn = NeuralNetwork([input_shape, 64, 10])
        print(f"[INFO] {nn}")
        nn.fit(X_train_dataset, y_train, epochs=10, displayUpdate=1)
        predictions = nn.predict(X_test_dataset)
        y_pred = predictions.argmax(axis=1)
        report = classification_report(y_test.argmax(axis=1), y_pred, target_names = labels)
        with open('out/nn_report_cifar10.txt', 'w') as file:
            file.write(report)
    elif data_set == "MNIST784":
        X, y = fetch_openml('mnist_784', return_X_y = True)
         
        X = np.array(X)
        y = np.array(y)
        
        classes = sorted(set(y))
        nclasses = len(classes)
         
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 7500, test_size = 2500)
         
        X_train_scaled = X_train/255
        X_test_scaled = X_test/255
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)
        print("[INFO] training network...")
        input_shape = X_train_scaled.shape[1]
        nn = NeuralNetwork([input_shape, 64, 10])
        print(f"[INFO] {nn}")
        nn.fit(X_train_scaled, y_train, epochs=10, displayUpdate=1)
        predictions = nn.predict(X_test_scaled)
        y_pred = predictions.argmax(axis=1)
        report = classification_report(y_test.argmax(axis=1), y_pred, target_names = classes)
        with open('out/nn_report_mnist.txt', 'w') as file:
            file.write(report)
            
    return X_train, y_train, X_test, y_test 

def parse_args(): 
    """
This function initializes the argument parser and defines which arguments to acquire from the terminal
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ds","--dataset",required=True, help = "The dataset to train neural network on")
    args = vars(ap.parse_args())
    return args 

    
def main():
    """
The main function defines which functions to run, when the script is run from the terminal, and which arguments to take. 
    """
    args = parse_args()
    X_train, y_train, X_test, y_test = nn_classifier(args["dataset"])


if __name__== "__main__":
    main()