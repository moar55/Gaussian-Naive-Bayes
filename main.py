import numpy as np
import math
from collections import defaultdict
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
import pandas as pd

class GaussianNB():
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.classes = set(Y)
        self.features = X.shape[1]
        self.learn_mean_and_variance()
        self.learn_prior_propabilities()
        del self.X
        del self.Y
    
    def learn_prior_propabilities(self):
        priors = defaultdict(int)
        for y in self.Y:
            priors[y] += 1

        priors_arr = np.array(list(priors.values()))
        priors_arr = np.divide(priors_arr,len(self.Y))
        self.priors = priors_arr
    
    def predict(self,X):
        predictions = []
        for i in range(X.shape[0]):
            max_prob = (0,-1)
            for y in self.classes:
                likelihood = []
                mean = self.mean_variance[y][0]
                variance = self.mean_variance[y][1]
                for ii in range(X.shape[1]):
                    if variance[ii] != 0:
                        a = 1/(math.sqrt(2 * math.pi * variance[ii]))
                        try :
                            b = math.exp( -( math.pow((X[i,ii] - mean[ii]),2) / (2 * variance[ii])) )
                            if a * b < 0.1:
                                likelihood.append(0.1)
                            else:
                                likelihood.append(a * b)
                        except OverflowError:
                            likelihood.append(0.1)
                    else:
                        likelihood.append(0.1)

                likelihood = np.prod(np.array(likelihood))
                prob = self.priors[y] * likelihood
                if prob > max_prob[0]:
                    max_prob = (prob,y)
            predictions.append(max_prob[1])
        
        return predictions
            

    def learn_mean_and_variance(self):
        mean_variance = {}
        for y in self.classes:
            y_features = []
            for i in range(self.X.shape[0]):
                if self.Y[i] == y:
                    y_features.append(self.X[i])
            
            mean_variance[y] = (np.mean(y_features,axis=0),  np.power(np.std(np.array(y_features),axis=0),2))

        self.mean_variance = mean_variance

if __name__ == "__main__":
    path = "Problem 2 Dataset/Train/"
    X = []
    Y = []
    for filename in os.listdir(path):
        img = imread(path+filename)
        X.append(img.flatten())
        Y.append(ord('z') - ord(filename[2]))

    gnb = GaussianNB(np.divide(np.array(X),255),np.array(Y))

    path = "Problem 2 Dataset/Test/"
    X = []
    Y = []
    for filename in os.listdir(path):
        img = imread(path+filename)
        X.append(img.flatten())
        Y.append(ord('z') - ord(filename[2]))

    results = gnb.predict(np.divide(np.array(X),255))
    print(results)
    print(Y)

    freq = defaultdict(int)

    total = 0
    for i in range(len(results)):
        if results[i] == Y[i]:
            total += 1
            freq[chr(ord('z') - Y[i])] += 1
    
    accuracy = (total)/len(results)
    print("{}% Accuracy".format(accuracy * 100))

    plt.bar(freq.keys(),freq.values())
    plt.title("Accuracy")
    plt.savefig("Accuracy.jpg")
