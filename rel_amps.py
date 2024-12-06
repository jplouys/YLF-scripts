"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2024-12-03 10:01:39
 Last Modification Date: 2024-12-06 15:16:26


"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

date = "24_12_06/1e-6/4"
fit_function = "three_peaks"
fit_results_location = date + "/" + fit_function


os.chdir(fit_results_location)
file_list = glob.glob("*.npy")

p, e, c = [], [], []

for file in file_list:
    params, errors = np.load(file)
    amp3 = params[5]
    error3 = errors[5]
    file.split("_")
    current = float(file.split("_")[1] + "." + file.split("_")[2])
    p.append(amp3)
    e.append(error3)
    c.append(current)

plt.errorbar(c, p, yerr=e, fmt="o")
plt.xlabel("Laser current [mA]")
plt.ylabel("Relative amplitude of peak 3 wrt peak 1")
plt.savefig("rel_amps.pdf", bbox_inches="tight")
