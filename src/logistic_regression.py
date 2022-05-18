"""
Firstly, let's import the libraries we'll be using for the script!
"""
# path tools
import argparse


# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


def log_reg(data_set):
    """
This function loads the dataset (either cifar-10 or mnist784) as input from the terminal, splits it into test and train data and scales the images. For CIFAR10 it further normalizes by turning the images into greyscale. It then uses logistic regression to make predections on the test data based on what it's learned on the train data. Finally it saves a classification report to the "out" folder. 
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
 
        clf = LogisticRegression(penalty = "none",
                                 tol = 0.1,
                                 solver = "saga",
                                 multi_class = "multinomial").fit(X_train_dataset, y_train)
        y_pred = clf.predict(X_test_dataset)
     
        report = classification_report(y_test, y_pred, target_names = labels)
     
        with open('out/lr_report_cifar10.txt', 'w') as file:
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
        clf = LogisticRegression(multi_class="multinomial").fit(X_train_scaled,y_train)
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = metrics.classification_report(y_test,y_pred)
        with open('out/lr_report_mnist.txt', 'w') as file:
            file.write(report)
        return report 

    
def parse_args():
    """
This function initializes the argument parser and defines which arguments to acquire from the terminal
    """
    ap = argparse.ArgumentParser() 
    ap.add_argument("-ds","--dataset",required=True, help = "The dataset to make logistic regression on. Must be either CIFAR10 or MNIST784")
    args = vars(ap.parse_args())
    return args 
                    
def main():
    """
The main function defines which functions to run, when the script is run from the terminal. 
    """
    args = parse_args()
    report = log_reg(args["dataset"])
    
    

if __name__== "__main__":
    main()