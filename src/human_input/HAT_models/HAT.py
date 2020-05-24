import pandas as pd
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import shutil
import glob

from NN import ConvNet
from sklearn_regressor import sklearn_regressor


class HAT:
    """
    encompassing class for the HAT algorithm. Was supposed to be more polished and nicely integrated with the
    Model class by means of Strategy design pattern, but this was not possible because of time constraints.
    """

    def __init__(self):
        # make the master training file
        self.import_data()

        # parse the .csv file to a Pandas dataframe
        self.df = pd.DataFrame(pd.read_csv("merged.csv"))

        X = self.df[["state"]].values.tolist()
        y = self.df["action"].values.tolist()

        # train-test split for the convolutional model
        train, test = sklearn.model_selection.train_test_split(self.df.values.tolist())

        # train-test split for the SGDRegressor model
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

        # # convolutional neural network model
        # model = ConvNet(train)
        #
        # model.train()

        # # SGDRegressor model
        # model = sklearn_regressor(X_train, y_train)
        #
        # model.train()

    def transform_csv(self, X_old):
        """function for transforming a csv to correct format, not needed in current implementation"""
        X = []

        for channel in X_old:
            channel = channel.split('\n')
            rows = []
            for row in channel:
                row = row.strip('][').split()
                if row[0] == '[':
                    del row[0]
                if row[0][0] == '[':
                    row[0] = row[0][1:]
                rows.append(list(map(int, row)))

            X.append(rows)

    def import_data(self):
        """function for combining training data from all per-episode files into 1 master file"""
        # import csv files from folder
        path = r'../training_data'
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


model = HAT()
