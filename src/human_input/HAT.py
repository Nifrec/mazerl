import pandas as pd
import sklearn
from sklearn.svm import SVC
import random
import spacy
import numpy as np


class Classifier:

    def __init__(self):
        df = pd.read_csv("../res/EN_NL_DATA.csv")

        self.nlp = spacy.load('en_core_web_sm')

        X = []
        y = []

        for entry in df.values:
            if not np.isnan(entry[1]):
                X.append(self.nlp(entry[0]).vector)
                y.append(entry[1])

        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

        # Create a support vector classifier
        self.clf = SVC(C=1, gamma='auto')

        # Fit the classifier using the training data
        self.clf.fit(X, y)

        # Predict the labels of the test set
        # y_pred = clf.predict(X_test)

        # # Count the number of correct predictions
        # n_correct = 0
        # for i in range(len(y_test)):
        #     if y_pred[i] == y_test[i]:
        #         n_correct += 1
        #
        # print("Predicted {0} correctly out of {1} test examples".format(n_correct, len(y_test)))

    def classify(self, message):
        i = [self.nlp(message).vector]
        o = int(self.clf.predict(i)[0])
        # print("Message is \"{}\", corresponding label is {}".format(message, output))
        return o