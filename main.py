"""
Perform MCR-ALS on XANES data normalized in Athena (.nor files)
TODO: add filter for non-physical results
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

from read_data import read_data


def main():
    # read in the data
    path = ''
    filename = 'CuXX_PAA50_and_Cu20_PAA50_NaCl-treatment_and_ISS-twin.nor'
    df, energies = read_data(path, filename, min_energy=8970, max_energy=9050, plot_data=True)

    # get S guess and D
    Data_df = df.filter(regex='PAMAM')
    Data = np.array(Data_df).T

    s_init_all = df.filter(regex=r'(standard|foil)')  # get standards/foil data and put into S guess

    # loop over all possible combinations of specified number of standards
    n_standards = s_init_all.keys().size - 2
    for standards in combinations(s_init_all, n_standards):

        s_init_df = s_init_all[list(standards)]
        s_init = np.array(s_init_df)

        # remove 'number' from standard names
        st_names = [name.split('.', 1)[0] for name in s_init_df.keys()]

        mcrar = McrAR(st_regr=NNLS(), c_regr=OLS(), c_constraints=[ConstraintNonneg(), ConstraintNorm()])
        # mcrar = McrAR(st_regr=NNLS(), c_regr=NNLS(), c_constraints=[ConstraintNorm()])
        mcrar.fit(Data, ST=s_init.T)

        # plot results
        plt.figure(figsize=[16, 12])

        ax1 = plt.subplot(321, title='Initial Spectra Guess')
        plt.plot(energies, s_init, label=st_names)
        plt.legend()

        ax2 = plt.subplot(322, sharey=ax1, title='MCR-AR Retrieved Spectra')
        plt.plot(energies, mcrar.ST_opt_.T)

        ax3 = plt.subplot(323, title='Data')
        plt.plot(energies, Data.T, label=Data_df.keys())
        plt.legend()

        ax4 = plt.subplot(324, sharey=ax3, title='MCR-AR Fit')
        D_opt_ = np.dot(mcrar.C_opt_, mcrar.ST_opt_)
        plt.plot(energies, D_opt_.T)

        ax5 = plt.subplot(325, title='Difference ($D$ - $D_{cacl}$)')
        plt.plot(energies, (Data - D_opt_).T)

        m = Data.shape[0]  # number of measurements
        ax6 = plt.subplot(326, title='Concentrations')
        plt.plot(range(m), mcrar.C_opt_)

        min_err = np.min(mcrar.err)
        plt.suptitle(f'min mse = {min_err}')
        plt.savefig(f'test_images/{n_standards}_standards/' + '_'.join(st_names))
        # plt.show()


if __name__ == '__main__':
    main()

