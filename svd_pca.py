"""
Compute and plot singular value decomposition of data matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from read_data import read_data

from sklearn.decomposition import PCA


def svd(a, x):
    # a = a.T
    # a = mean_center(a)
    # a = a.T
    u, s, vh = np.linalg.svd(a)
    print(a.shape, x.shape)

    print(u.shape, s.shape, vh.shape)
    print(s)

    # loop over col of u
    for i, ui in enumerate(u.T):
        if i < vh.shape[0]:
            vi = vh[i]

            fig, axs = plt.subplots(1, 2)
            plt.suptitle(f'$s_{i}$ = {s[i]}')

            axs[0].plot(x, ui)
            axs[1].plot(range(vh.shape[0]), vi)

            plt.show()


def mean_center(a):
    a_mean = a.mean(axis=0)  # col means
    return a - a_mean


def pca_svd(a, x):
    a = mean_center(a)
    # plt.plot(a.T)
    # plt.show()
    u, s, vh = np.linalg.svd(a)
    n = s.size
    plt.plot(x, vh[0])  # first principal direction
    plt.show()
    pc = u @ np.diag(s)

    lam = (s**2)/(n - 1)
    print(lam)
    plt.title('scree plot')
    plt.plot(range(1, n+1), lam)
    plt.show()


in_path = ''
filename = 'CuXX_PAA50_and_Cu20_PAA50_NaCl-treatment_and_ISS-twin.nor'
df, energies = read_data(in_path, filename, min_energy=8970, max_energy=9050, plot_data=False)

Data_df = df.filter(regex='PAMAM')
Data = np.array(Data_df)

# svd(Data, energies)  # SVD prefers data in col vectors
pca_svd(Data.T, energies)  # PCA prefers data in row vectors

# # using sklearn PCA
# pca = PCA()
# pca.fit(Data.T)
# print(pca.explained_variance_)
# plt.plot(energies, pca.components_[0])
# plt.show()
