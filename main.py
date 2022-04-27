"""
Perform MCR-ALS on XANES data normalized in Athena (.nor files)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymcr
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm


def main():
    # read in the data
    path = ''
    filename = 'Cu-b26-9_CRS_Cu20_PAMAM_NaCl_w-standards_cal_nor.nor'
    df = pd.read_table(path + filename, delimiter='\s+', header=15)
    df = df.shift(periods=1, axis="columns")  # shift data due to '#' column from .nor

    # cut to desired energy range
    min_energy, max_energy = 8970, 9050
    df = df[df['energy'].between(min_energy, max_energy)]
    energies = np.array(df['energy'])

    # plot data
    for col in df:
        if col != '#' and col != 'energy':
            plt.plot(energies, df[col], label=col)
    plt.legend()
    plt.xlim(min_energy, max_energy)
    # plt.show()

    # get ST guess and D
    s_init_df = df.filter(regex=r'(standard|foil)')  # get standards/foil data and put into S guess
    s_init_df = s_init_df.drop(columns=['CuF2_standard.001', 'CuO_standard.001'])
    s_init = np.array(s_init_df)
    D_df = df.filter(regex='Cu20_PAMAM')
    D = np.array(D_df).T

    mcrar = McrAR(st_regr=OLS(), c_regr=OLS(), c_constraints=[ConstraintNonneg(), ConstraintNorm()])
    # mcrar = McrAR(st_regr=NNLS(), c_regr=NNLS(), c_constraints=[ConstraintNorm()])
    mcrar.fit(D, ST=s_init.T, verbose=True)

    # plot results
    plt.clf()
    ax1 = plt.subplot(121, title='Initial Spectra Guess')
    plt.plot(energies, s_init, label=s_init_df.keys())
    plt.legend()

    ax2 = plt.subplot(122, sharey=ax1, title='MCR-AR Retrieved Spectra')
    plt.plot(energies, mcrar.ST_opt_.T)
    plt.show()

    plt.clf()
    ax3 = plt.subplot(121, title='Data')
    plt.plot(energies, D.T, label=D_df.keys())
    plt.legend()

    ax4 = plt.subplot(122, sharey=ax3, title='MCR-AR Fit')
    D_opt_ = np.dot(mcrar.C_opt_, mcrar.ST_opt_)
    plt.plot(energies, D_opt_.T)
    plt.show()


if __name__ == '__main__':
    main()

