"""
Read in XANES data normalized in Athena for use in MCR-ALS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(path, fn, min_energy, max_energy, plot_data=False):
    # get data into dataframe
    df = pd.read_table(path + fn, delimiter='\s+', header=15)
    df = df.shift(periods=1, axis="columns")

    # cut to desired energy range
    df = df[df['energy'].between(min_energy, max_energy)]
    energies = np.array(df['energy'])

    # plot data
    if plot_data:
        for col in df:
            if col != '#' and col != 'energy':
                plt.plot(energies, df[col], label=col)
        plt.legend()
        plt.xlim(min_energy, max_energy)
        plt.show()

    return df, energies
