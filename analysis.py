import numpy as np
import matplotlib.pyplot as plt  # noqa
from fit import fit
import glob
import os
from tqdm import tqdm as pbar
from colorama import Fore


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
f3 = 992.6
f4 = 980.6
f5 = 983.5
f6 = 995.5

A = 1600
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

# p0 = [
#     f1,
#     f2,
#     f3,
#     f4,
#     f5,
#     f6,
#     A,
#     Arel2,
#     Arel3,
#     Arel4,
#     Arel5,
#     Arel6,
#     gamma1,
#     gamma2,
#     gamma3,
#     gamma4,
#     gamma5,
#     gamma6,
#     250,
# ]
# names = [
#     "Peak 1",
#     "Peak 2",
#     "Peak 3",
#     "Peak 4",
#     "Peak 5",
#     "Peak 6",
#     "Amplitude",
#     "Relative amplitude 2",
#     "Relative amplitude 3",
#     "Relative amplitude 4",
#     "Relative amplitude 5",
#     "Relative amplitude 6",
#     "HWHM 1",
#     "HWHM 2",
#     "HWHM 3",
#     "HWHM 4",
#     "HWHM 5",
#     "HWHM 6",
#     "Offset",
# ]
# bounds = (
#     [
#         f1 - 5,
#         f2 - 5,
#         f3 - 5,
#         f4 - 5,
#         f5 - 5,
#         f6 - 5,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#     ],
#     [
#         f1 + 5,
#         f2 + 5,
#         f3 + 5,
#         f4 + 5,
#         f5 + 5,
#         f6 + 5,
#         5000,
#         5,
#         5,
#         5,
#         5,
#         5,
#         100,
#         100,
#         100,
#         100,
#         100,
#         100,
#         1000,
#     ],
# )

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

# * Four peaks

# p0 = [f1, f2, f3, f4, A, Arel2, Arel3, Arel4, gamma1, gamma2, gamma3, gamma4, 250]
# names = [
#     "Peak 1",
#     "Peak 2",
#     "Peak 3",
#     "Peak 4",
#     "Amplitude",
#     "Relative amplitude 2",
#     "Relative amplitude 3",
#     "Relative amplitude 4",
#     "HWHM 1",
#     "HWHM 2",
#     "HWHM 3",
#     "HWHM 4",
#     "Offset",
# ]
# bounds = (
#     [f1 - 5, f2 - 5, f3 - 5, f4 - 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [f1 + 5, f2 + 5, f3 + 5, f4 + 5, 5000, 5, 5, 5, 100, 100, 100, 100, 1000],
# )

# * Three peaks

p0 = [f1, f2, f3, A, Arel2, Arel3, gamma1, gamma2, gamma3, 250]
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
    [f1 - 2, f2 - 5, f3 - 5, 0, 0, 0, 0, 0, 0, 0],
    [f1 + 2, f2 + 5, f3 + 5, 5000, 5, 5, 100, 100, 100, 1000],
)

# %% General run

data_location = "24_12_04"
fit_results_location = "three_peaks"
os.chdir(data_location)
file_list = glob.glob("*.asc")
if not os.path.exists(fit_results_location):
    os.makedirs(fit_results_location)


for dataset_name in pbar(file_list, desc="Fitting", colour="green"):
    data = np.loadtxt(dataset_name, skiprows=32, delimiter=",")
    f_raw = data[:, 0]
    a = np.where(f_raw > 950)[0][0]
    b = np.where(f_raw < 1000)[0][-1]
    f = f_raw[a:b]
    s = data[a:b, 1]

    f, s = remove_noise_peaks(f, s)

    norm_counts = s[np.argmin(abs(f - 960.1))]
    s /= norm_counts
    try:
        fit_results = fit(
            three_peaks,
            f,
            s,
            xtitle="$\lambda$ [nm]",
            ytitle="Normalized counts",
            title=fit_results_location + "/" + dataset_name[:-4],
            guess=p0,
            nombre_params=names,
            msize=5,
            legend=False,
            # bounds=bounds,
            silent=0,
        )
        params, errors = fit_results[0], fit_results[1]

        np.save(
            fit_results_location + "/fit_" + dataset_name[:-4],
            [params, errors],
        )
    except:
        print(Fore.RED + "Fitting failed for", dataset_name, Fore.RESET)
