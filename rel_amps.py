"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2024-12-03 10:01:39
 Last Modification Date: 2024-12-05 10:33:38


"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

date = "24_12_04"
fit_function = "three_peaks"
n_peaks = 3
fit_results_location = date + "/" + fit_function


os.chdir(fit_results_location)
file_list = glob.glob("*.npy")

p, e, c = [], [], []

for file in file_list:
    params, errors = np.load(file)
    amp3 = params[n_peaks + 2]
    error3 = errors[n_peaks + 2]
    file.split("_")
    current = float(file.split("_")[1] + "." + file.split("_")[2])
    p.append(amp3)
    e.append(error3)
    c.append(current)

plt.errorbar(c, p, yerr=e, fmt="o")
plt.xlabel("Laser current [mA]")
plt.ylabel("Relative amplitude of peak 3 wrt peak 1")
plt.savefig("rel_amps.pdf", bbox_inches="tight")
