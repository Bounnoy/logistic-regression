# Bounnoy Phanthavong (ID: 973081923)
# Homework 5
#
# This is a machine learning program that uses logistic regression
# to classify spam.
#
# This program was built in Python 3.

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv

# This function creates a confusion matrix and outputs results
# to the console and to a 'results.csv' file.
def confuse(predict, actual, accuracy):
    unique, _ = np.unique(actual, return_counts = True)
    matrix = np.zeros((len(unique), len(unique)))

    # Plot results of our classifier into matrix.
    for i in range(len(predict)):
        matrix[ int(predict[i]) ][ int(actual[i]) ] += 1

    np.set_printoptions(suppress = True)
    print("\nConfusion Matrix")
    print(matrix, "\n")

    with open('results.csv', 'a') as csvFile:
        w = csv.writer(csvFile)
        w.writerow([])
        w.writerow(["Confusion Matrix"])
        for k in range(len(unique)):
            w.writerow(matrix[k,:])
        w.writerow(["Final Accuracy"] + [accuracy])
        w.writerow([])

if __name__ == '__main__':

    # Load data.
    fName = "spambase/spambase.data"
    fileTrain = Path(fName)

    if not fileTrain.exists():
        sys.exit(fName + " not found")

    trainData = np.genfromtxt(fName, delimiter=",")

    # Split input data into X and labels into Y.
    X = trainData[:,0:-1]
    Y = trainData[:,-1]

    # Split X and Y by 50% for training and testing.
    # Keep proportion of labels same across both sets.
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.5, stratify = Y)

    # Scale data using standardization.
    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    # Run Logistic Regression learner on training data and then test learned model on test data.
    classify = LogisticRegression()
    classify.fit(XTrain, YTrain)
    YPredict = classify.predict(XTest)

    # Report accuracy, precision, and recall.
    accuracy = accuracy_score(YTest, YPredict)
    precision = precision_score(YTest, YPredict)
    recall = recall_score(YTest, YPredict)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    confuse(YPredict, YTest, accuracy)
