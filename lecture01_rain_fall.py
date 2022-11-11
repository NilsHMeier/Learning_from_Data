# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:19:07 2020
Learning from Data - Lecture 1
Analysing the Australia Rain dataset to predict rainfall for the following day
@author: Learning from Data team
"""

# Load required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing


def import_preproc(filename):
    """Import data and preprocess."""
    # Source: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
    weather = pd.read_csv(filename)

    # Remove the target variable for the amount of rainfall
    weather = weather.drop(columns=['RISK_MM'])

    # Make the target variable machine-readable
    weather.RainToday.replace(('Yes', 'No'), (1, 0), inplace=True)
    weather.RainTomorrow.replace(('Yes', 'No'), (1, 0), inplace=True)

    # NA treatment - remove rows with NA values
    weather = weather.dropna(axis=0)

    # Only keep the continuous variables
    weather = weather.select_dtypes(exclude=['object'])

    # Create the input and output variables from the dataset
    # Remove the binary variable for rain on the previous day,
    # as we already have a variable with the amount of rainfall
    # on the previous day
    x = weather.drop(columns=['RainTomorrow'])
    y = weather.RainTomorrow

    # Scale the input
    x = preprocessing.scale(x)

    return x, y


def plot_roc():
    """Plot the ROC Curve."""
    # Predict values for y (probabilities)
    y_pred_prob = classifier.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    # Plot the ROC curve
    fig, axs = plt.subplots()
    axs.plot([0, 1], [0, 1], '--', label='No Skill')
    axs.plot(fpr, tpr, '.-', label='Logistic Regression, auc= %.3f' % auc)
    axs.set(xlabel='False Positive Rate', ylabel='True Positive Rate', )
    axs.legend(loc=4)
    fig.savefig('Figures/ROC_Curve.png')
    plt.show()


if __name__ == '__main__':
    x, y = import_preproc('data/lecture01_weatherAUS.csv')

    # Create train-test dataset split at 70% and 30% and returning it
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=2)

    # Use logistic regression and train it on the training dataset
    classifier = LogisticRegression(random_state=0, max_iter=1000). \
        fit(x_train, y_train)

    # Predict labels of our model on the test data
    pred_label = classifier.predict(x_test)

    # Check the accuracy of our model - predicted vs true labels
    print(f'Accuracy Score of Logistic Regression: '
          f'{accuracy_score(y_test, pred_label)}')

    plot_roc()
