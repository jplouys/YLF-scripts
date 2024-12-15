"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2024-12-13 14:01:22
 Last Modification Date: 2024-12-13 14:20:23

Another File Header is a Visual Studio Code extension to automatically or by command insert a header to your files.

"""

import numpy as np
import matplotlib.pyplot as plt  # noqa
from fit import fit
import glob
import os
from tqdm import tqdm as pbar
from colorama import Fore
import shutil


def lorentzian(x, x0, gamma):
    return 1 / np.pi * gamma / ((x - x0) ** 2 + gamma**2)  # gamma is the HWHM


def six_peaks_fixed(
    f,
    f0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
    gamma5,
    gamma6,
    offset,
):
    f_peaks = [960.06, 971.53, 980.58, 983.48, 992.56, 995.52]
    return (
        A1 * lorentzian(f - f0, f_peaks[0], gamma1)
        + A2 * lorentzian(f - f0, f_peaks[1], gamma2)
        + A3 * lorentzian(f - f0, f_peaks[2], gamma3)
        + A4 * lorentzian(f - f0, f_peaks[3], gamma4)
        + A5 * lorentzian(f - f0, f_peaks[4], gamma5)
        + A6 * lorentzian(f - f0, f_peaks[5], gamma6)
        + offset
    )


def remove_noise_peaks(f, s, threshold=50):
    d = abs(np.diff(s)) > threshold
    d = np.append(d, False)
    d_copy = d.copy()
    i = 1
    while i < len(d):
        if d[i]:
            d_copy[i - 1] = True
            d_copy[i + 1] = True
        else:
            pass
        i += 1
    return f[True ^ d_copy], s[True ^ d_copy]


p0 = [
    -0.8814403698024486,
    12.610031946156209,
    0,
    121.51365520361118,
    0,
    9.550582946877457,
    1.572231110343819,
    7.245674762481448,
    19.529988221960217,
    17.163723490468797,
    12.239875062216496,
    6.316116999372291,
    2.739584157972279,
    0.25846303080229394,
]


# plt.plot(f, s)
# plt.plot(f, fit_function(f, *p0))
# plt.show()

names = [
    "Frequency calibration offset",
    "Amplitude 1",
    "Amplitude 2",
    "Amplitude 3",
    "Amplitude 4",
    "Amplitude 5",
    "Amplitude 6",
    "HWHM 1",
    "HWHM 2",
    "HWHM 3",
    "HWHM 4",
    "HWHM 5",
    "HWHM 6",
    "Offset",
]
bounds = (
    [
        -2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    [
        2,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        30,
        30,
        30,
        30,
        30,
        30,
        1,
    ],
)

six_peaks_fixed_settings = six_peaks_fixed, p0, names, bounds

location = "24_12_05/1e-6"
if not os.path.exists(location + "/data"):
    os.makedirs(location + "/data")
# Move .asc files to data folder
os.chdir(location)
file_list = glob.glob("*.asc")
for file_name in file_list:
    if file_name.endswith(".asc"):
        shutil.move(file_name, "data/" + file_name)
os.chdir("../../")
fit_results_location = "six_peaks_fixed"
fit_function, p0, names, bounds = six_peaks_fixed_settings

data_location = location + "/data"
os.chdir(data_location)
file_list = glob.glob("*.asc")
os.chdir("../")
if not os.path.exists(fit_results_location):
    os.makedirs(fit_results_location)


error_log = "\n"

for dataset_name in pbar(file_list, desc="Fitting", colour="green"):
    data = np.loadtxt("data/" + dataset_name, skiprows=32, delimiter=",")
    f_raw = data[:, 0]
    a = np.where(f_raw > 955)[0][0]
    b = np.where(f_raw < 1000)[0][-1]
    f = f_raw[a:b]
    s = data[a:b, 1]

    f, s = remove_noise_peaks(f, s)

    norm_counts = s[np.argmin(abs(f - 960.1))]
    s /= norm_counts
    try:
        fit_results = fit(
            fit_function,
            f,
            s,
            xtitle="$\lambda$ [nm]",
            ytitle="Normalized counts",
            title=fit_results_location + "/" + dataset_name[:-4],
            guess=p0,
            nombre_params=names,
            msize=5,
            legend=False,
            bounds=bounds,
            silent=1,
        )
        params, errors = fit_results[0], fit_results[1]

        np.save(
            fit_results_location + "/fit_" + dataset_name[:-4],
            [params, errors],
        )
    except:  # noqa: E722
        error_log += "Fitting failed for " + str(dataset_name) + "\n"
print(Fore.RED + error_log + Fore.RESET)
