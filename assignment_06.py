import numpy as np
import matplotlib.pyplot as plt


def calculate_mh(n: np.ndarray) -> np.ndarray:
    return 0.5 * (n ** 2) + 0.5 * n + 1


def plot_vc_bound(epsilon: float = 0.1) -> None:
    # Set a range of x values
    N = np.arange(0, 25000, dtype=np.float64)

    # Calculate the VC bound
    vc_bound = 4 * calculate_mh(2 * N) * np.exp((-1 / 8) * (epsilon ** 2) * N)

    # Determine the first n where the VC bound is smaller than 1
    n = np.argmax(vc_bound < 1)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(N, vc_bound, label='VC Bound')
    ax.plot(N, np.ones(N.shape), 'r--', label='1')
    ax.set(title=f'VC Bound below 1 for {n=}', xlabel='n', ylabel='VC Bound',
           ylim=(0, 2), xlim=(0, 25000))
    ax.legend()
    fig.savefig('Figures/assignment_06_VC_Bound.png')
    plt.show()


if __name__ == '__main__':
    plot_vc_bound()
