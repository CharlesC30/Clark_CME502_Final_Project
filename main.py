"""
Perform MCR-ALS on XANES data normalized using Athena (.nor files)
This program uses the pyMCR package (available here: https://pages.nist.gov/pyMCR/installing.html)

To run the code specify your input and output directory paths using the 'in_path' and 'out_path' variables in the
main function.
Specify the file you would like to run MCR on using the 'filename' variable.
For testing multiple standard combinations the 'n_standards' variable may be edited.

Results will be saved in a new directory created in the output path.
The directory will be labeled with the number of standards used for the initial guess and results will be separated
based on physical validity.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from os.path import exists
from os import makedirs

from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

from read_data import read_data


def main():
    # read in the data
    in_path = ''
    filename = 'CuXX_PAA50_and_Cu20_PAA50_NaCl-treatment.nor'
    df, energies = read_data(in_path, filename, min_energy=8970, max_energy=9050, plot_data=True)

    out_path = 'CuXX_and_Cu20_NaCl_treatment'

    # get S guess and D
    Data_df = df.filter(regex='PAMAM')
    Data = np.array(Data_df).T

    s_init_all = df.filter(regex=r'(standard|foil)')  # get standards/foil data and put into S guess

    # loop over all possible combinations of specified number of standards
    all_standards = s_init_all.keys().size
    n_standards = all_standards - 2
    for standards in combinations(s_init_all, n_standards):

        s_init_df = s_init_all[list(standards)]
        s_init = np.array(s_init_df)

        # remove 'number' from standard names
        st_names = [name.split('.', 1)[0] for name in s_init_df.keys()]

        # perform MCR-ALS
        mcrar = McrAR(st_regr=NNLS(), c_regr=OLS(), c_constraints=[ConstraintNonneg(), ConstraintNorm()])
        mcrar.fit(Data, ST=s_init.T)

        # plot results
        plt.figure(figsize=[16, 15])

        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('axes', titlesize=18)
        plt.rc('figure', titlesize=21)

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

        # check if result is physically valid
        physical = True
        fitted_standard_avgs = np.mean(mcrar.ST_opt_, axis=1)
        if (fitted_standard_avgs < 0.5).any() or (fitted_standard_avgs > 2.0).any():
            physical = False

        # check if the output path exists, and if not create it
        if not exists(f'{out_path}/{n_standards}_standards'):
            makedirs(f'{out_path}/{n_standards}_standards/physical')
            makedirs(f'{out_path}/{n_standards}_standards/non-physical')

        # save results
        if physical:
            plt.savefig(f'{out_path}/{n_standards}_standards/physical/' + '_'.join(st_names))
        else:
            plt.savefig(f'{out_path}/{n_standards}_standards/non-physical/' + '_'.join(st_names))
        # plt.show()


if __name__ == '__main__':
    main()

