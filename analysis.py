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


def three_peaks(f, f1, f2, f3, A, Arel2, Arel3, gamma1, gamma2, gamma3, offset):
    return (
        A
        * (
            lorentzian(f, f1, gamma1)
            + Arel2 * lorentzian(f, f2, gamma2)
            + Arel3 * lorentzian(f, f3, gamma3)
        )
        + offset
    )


def four_peaks(
    f, f1, f2, f3, f4, A, Arel2, Arel3, Arel4, gamma1, gamma2, gamma3, gamma4, offset
):
    return (
        A
        * (
            lorentzian(f, f1, gamma1)
            + Arel2 * lorentzian(f, f2, gamma2)
            + Arel3 * lorentzian(f, f3, gamma3)
            + Arel4 * lorentzian(f, f4, gamma4)
        )
        + offset
    )


def five_peaks(
    f,
    f1,
    f2,
    f3,
    f4,
    f5,
    A,
    Arel2,
    Arel3,
    Arel4,
    Arel5,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
    gamma5,
    offset,
):
    return (
        A
        * (
            lorentzian(f, f1, gamma1)
            + Arel2 * lorentzian(f, f2, gamma2)
            + Arel3 * lorentzian(f, f3, gamma3)
            + Arel4 * lorentzian(f, f4, gamma4)
            + Arel5 * lorentzian(f, f5, gamma5)
        )
        + offset
    )


def six_peaks(
    f,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    A,
    Arel2,
    Arel3,
    Arel4,
    Arel5,
    Arel6,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
    gamma5,
    gamma6,
    offset,
):
    return (
        A
        * (
            lorentzian(f, f1, gamma1)
            + Arel2 * lorentzian(f, f2, gamma2)
            + Arel3 * lorentzian(f, f3, gamma3)
            + Arel4 * lorentzian(f, f4, gamma4)
            + Arel5 * lorentzian(f, f5, gamma5)
            + Arel6 * lorentzian(f, f6, gamma6)
        )
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


f1 = 960.1
f2 = 971.5
f2 = 977
f3 = 992.6
f4 = 980.6
f5 = 983.5
f6 = 995.5

A = 5
Arel2 = 2
Arel3 = 3
Arel4 = 0.5
Arel5 = 0.5
Arel6 = 3

gamma1 = 5
gamma2 = 10
gamma3 = 7
gamma4 = 10
gamma5 = 10
gamma6 = 10

# * Six peaks

p0 = [
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    A,
    Arel2,
    Arel3,
    Arel4,
    Arel5,
    Arel6,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
    gamma5,
    gamma6,
    250,
]
names = [
    "Peak 1",
    "Peak 2",
    "Peak 3",
    "Peak 4",
    "Peak 5",
    "Peak 6",
    "Amplitude",
    "Relative amplitude 2",
    "Relative amplitude 3",
    "Relative amplitude 4",
    "Relative amplitude 5",
    "Relative amplitude 6",
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
        f1 - 5,
        f2 - 5,
        f3 - 5,
        f4 - 5,
        f5 - 5,
        f6 - 5,
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
        f1 + 5,
        f2 + 5,
        f3 + 5,
        f4 + 5,
        f5 + 5,
        f6 + 5,
        5000,
        5,
        5,
        5,
        5,
        5,
        100,
        100,
        100,
        100,
        100,
        100,
        1000,
    ],
)

six_peaks_settings = six_peaks, p0, names, bounds

# * Five peaks

p0 = [
    f1,
    f2,
    f3,
    f4,
    f5,
    A,
    Arel2,
    Arel3,
    Arel4,
    Arel5,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
    gamma5,
    250,
]
names = [
    "Peak 1",
    "Peak 2",
    "Peak 3",
    "Peak 4",
    "Peak 5",
    "Amplitude",
    "Relative amplitude 2",
    "Relative amplitude 3",
    "Relative amplitude 4",
    "Relative amplitude 5",
    "HWHM 1",
    "HWHM 2",
    "HWHM 3",
    "HWHM 4",
    "HWHM 5",
    "Offset",
]
bounds = (
    [
        f1 - 5,
        f2 - 5,
        f3 - 5,
        f4 - 5,
        f5 - 5,
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
        f1 + 5,
        f2 + 5,
        f3 + 5,
        f4 + 5,
        f5 + 5,
        5000,
        5,
        5,
        5,
        5,
        100,
        100,
        100,
        100,
        100,
        1000,
    ],
)
five_peaks_settings = five_peaks, p0, names, bounds

# * Four peaks

p0 = [f1, f2, f3, f4, A, Arel2, Arel3, Arel4, gamma1, gamma2, gamma3, gamma4, 250]
names = [
    "Peak 1",
    "Peak 2",
    "Peak 3",
    "Peak 4",
    "Amplitude",
    "Relative amplitude 2",
    "Relative amplitude 3",
    "Relative amplitude 4",
    "HWHM 1",
    "HWHM 2",
    "HWHM 3",
    "HWHM 4",
    "Offset",
]
bounds = (
    [f1 - 2, f2 - 2, f3 - 2, f4 - 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [f1 + 2, f2 + 2, f3 + 2, f4 + 2, 5000, 5, 5, 5, 100, 100, 100, 100, 10000],
)
four_peaks_settings = four_peaks, p0, names, bounds
# * Three peaks

p0 = [f1, f2, f3, A, Arel2, Arel3, gamma1, gamma2, gamma3, 0]
names = [
    "Peak 1",
    "Peak 2",
    "Peak 3",
    "Amplitude",
    "Relative amplitude 2",
    "Relative amplitude 3",
    "HWHM 1",
    "HWHM 2",
    "HWHM 3",
    "Offset",
]
bounds = (
    [f1 - 5, f2 - 8, f3 - 5, 0, 0, 0, 0, 0, 0, 0],
    [f1 + 5, f2 + 5, f3 + 5, 5000, 5, 5, 100, 100, 100, np.inf],
)
three_peaks_settings = three_peaks, p0, names, bounds

# %% General run

location = "24_12_09/1e-6/1"
if not os.path.exists(location + "/data"):
    os.makedirs(location + "/data")
# Move .asc files to data folder
os.chdir(location)
file_list = glob.glob("*.asc")
for file_name in file_list:
    if file_name.endswith(".asc"):
        shutil.move(file_name, "data/" + file_name)
os.chdir("../../../")
fit_results_location = "three_peaks"
fit_function, p0, names, bounds = three_peaks_settings

data_location = location + "/data"
os.chdir(data_location)
file_list = glob.glob("*.asc")
os.chdir("../")
if not os.path.exists(fit_results_location):
    os.makedirs(fit_results_location)

l = "\n"
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
    except:
        l += "Fitting failed for " + str(dataset_name) + "\n"
print(Fore.RED + l + Fore.RESET)
