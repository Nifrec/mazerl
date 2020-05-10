import pandas as pd
import sklearn
from sklearn.svm import SVC
import numpy as np
import shutil
import glob


class Classifier:

    def __init__(self):
        self.import_data()

        self.data = pd.read_csv("merged.csv")

        X = []
        y = []

        for entry in self.data.values:
            X.append(entry[0])
            y.append(entry[1])

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

        # Create a support vector classifier
        self.clf = SVC(C=1, gamma='auto')

        # Fit the classifier using the training data
        self.clf.fit(X, y)

        # Predict the labels of the test set
        y_pred = self.clf.predict(X_test)

        # Count the number of correct predictions
        n_correct = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                n_correct += 1

        print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))

    def import_data(self):
        # import csv files from folder
        path = r'./training_data'
        allFiles = glob.glob(path + "/*.csv")
        allFiles.sort()  # glob lacks reliable ordering, so impose your own if output order matters
        with open('merged.csv', 'wb') as outfile:
            for i, fname in enumerate(allFiles):
                with open(fname, 'rb') as infile:
                    if i != 0:
                        infile.readline()  # Throw away header on all but first file
                    # Block copy rest of file from input to output without parsing
                    shutil.copyfileobj(infile, outfile)
                    print(fname + " has been imported.")

    def classify(self, state):
        output = self.clf.predict(state)[0]
        # print("Message is \"{}\", corresponding label is {}".format(message, output))
        return output


clas = Classifier()
