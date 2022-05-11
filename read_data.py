"""
Read in XANES data normalized in Athena (.nor files) for use in MCR-ALS
TODD: add data from materials project (https://materialsproject.org/)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(path, fn, min_energy, max_energy, plot_data=False):

    # loop over lines to find header
    header_line = 0
    with open(path + fn, 'r') as data_file:
        for i, line in enumerate(data_file):
            # in .nor files many -'s are used in line before header
            if '-----' in line:
                header_line = i + 1
                break

    # get data into dataframe
    df = pd.read_table(path + fn, delimiter='\s+', header=header_line)
    df = df.shift(periods=1, axis="columns")

    # cut to XANES energy range
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
