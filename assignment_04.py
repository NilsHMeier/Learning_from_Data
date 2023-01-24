import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def task_1():
    # Set up values
    x_values = np.linspace(-5, 5, 100)
    y_values = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x_values, y_values)

    # Create the function to determine the z-values
    def determine_class(x1, x2):
        if x1 >= 1:
            if x2 >= 2:
                if x1 >= 3:
                    return 1
                else:
                    return -1
            else:
                return -1
        else:
            if x2 < -1:
                return 1
            else:
                return -1

    # Determine the z-values
    z_values = np.vectorize(determine_class)(x, y).reshape(x.shape)

    # Plot the data
    fig, ax = plt.subplots()
    ax.contourf(x, y, z_values, cmap='RdBu', alpha=0.5)
    ax.set(title='Decision Boundaries', xlabel='x1', ylabel='x2',
           xlim=(min(x_values), max(x_values)), ylim=(min(y_values), max(y_values)))
    plt.show()
    plt.tight_layout()
    fig.savefig('Figures/assignment_04_Boundaries.png')
    plt.show()


def task_3():
    # Create the dataset from the exercise
    dataset = pd.DataFrame({'Instance': [1, 2, 3, 4, 5, 6, 7, 8],
                            'Outlook': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy'],
                            'Temperature': ['Warm', 'Warm', 'Mild', 'Warm', 'Cool', 'Cool', 'Warm', 'Mild'],
                            'Windy': [0, 1, 1, 1, 1, 0, 0, 1],
                            'Classification': ['+', '+', '+', '+', '-', '-', '-', '+']})

    # Calculate the entropy of the classification
    classification_entropy = stats.entropy(dataset['Classification'].value_counts(normalize=True), base=2)
    print(f'Entropy of the classification: {classification_entropy:.3f}')


def main():
    task_1()
    task_3()


if __name__ == '__main__':
    main()
