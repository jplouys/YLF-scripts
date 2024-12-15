"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2024-12-03 10:01:39
 Last Modification Date: 2024-12-13 14:46:54


"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from fit import propagador

date = "24_12_05/1e-6"
fit_function = "three_peaks_separate_fits"
fit_results_location = date + "/" + fit_function
peaks_location = [3, 5]


def division(a, b):
    return b / a


os.chdir(fit_results_location)
file_list = glob.glob("*.npy")

p, e, c = [], [], []

for file in file_list:
    params, errors = np.load(file)
    peak1 = params[peaks_location[0]]
    error_peak1 = errors[peaks_location[0]]
    peak2 = params[peaks_location[1]]
    error_peak2 = errors[peaks_location[1]]
    amp3, error3 = propagador(division, [peak1, peak2], [error_peak1, error_peak2])
    file.split("_")
    current = float(file.split("_")[1] + "." + file.split("_")[2])
    print(amp3, error3, current)
    p.append(amp3)
    e.append(error3)
    c.append(current)

plt.errorbar(c, p, yerr=e, fmt="o")
plt.xlabel("Laser current [mA]")
plt.ylabel("Relative amplitude of peak 3 wrt peak 1")
plt.savefig("rel_amps.pdf", bbox_inches="tight")
