"""
Perform MCR-ALS on XANES data normalized in Athena (.nor files)
"""

import numpy as np
import matplotlib.pyplot as plt

from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

from read_data import read_data


def main():
    # read in the data
    path = ''
    filename = 'Cu-b26-9_CRS_Cu20_PAMAM_NaCl_w-standards_cal_nor.nor'
    df, energies = read_data(path, filename, min_energy=8970, max_energy=9050, plot_data=False)

    # get ST guess and D
    s_init_df = df.filter(regex=r'(standard|foil)')  # get standards/foil data and put into S guess
    s_init_df = s_init_df.drop(columns=['CuF2_standard.001', 'CuO_standard.001'])
    s_init = np.array(s_init_df)
    D_df = df.filter(regex='merge')
    D = np.array(D_df).T

    mcrar = McrAR(st_regr=NNLS(), c_regr=OLS(), c_constraints=[ConstraintNonneg(), ConstraintNorm()])
    # mcrar = McrAR(st_regr=NNLS(), c_regr=NNLS(), c_constraints=[ConstraintNorm()])
    mcrar.fit(D, ST=s_init.T, verbose=True)

    # plot results
    plt.clf()
    ax1 = plt.subplot(321, title='Initial Spectra Guess')
    plt.plot(energies, s_init, label=s_init_df.keys())
    plt.legend()

    ax2 = plt.subplot(322, sharey=ax1, title='MCR-AR Retrieved Spectra')
    plt.plot(energies, mcrar.ST_opt_.T)

    ax3 = plt.subplot(323, title='Data')
    plt.plot(energies, D.T, label=D_df.keys())
    plt.legend()

    ax4 = plt.subplot(324, sharey=ax3, title='MCR-AR Fit')
    D_opt_ = np.dot(mcrar.C_opt_, mcrar.ST_opt_)
    plt.plot(energies, D_opt_.T)

    m = D_opt_.shape[0]  # number of measurements
    ax5 = plt.subplot(325, title='Concentrations')
    plt.plot(range(m), mcrar.C_opt_)
    plt.xticks(ticks=range(m), labels=np.arange(0, 91, 30))
    plt.xlabel('NaCl treatment time (min)')
    plt.show()


if __name__ == '__main__':
    main()

