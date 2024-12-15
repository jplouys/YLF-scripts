"""

Author: Jean Paul
Email: jean.louys-sanso@uibk.ac.at

Creation Date: 2024-12-15 22:45:26
 Last Modification Date: 2024-12-15 22:53:41

Another File Header is a Visual Studio Code extension to automatically or by command insert a header to your files.

"""

import numpy as np
import matplotlib.pyplot as plt  # noqa


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


location = "24_12_13/2/"
prefix = "100_sec_"

plt.figure()

for i in ["000", "045", "090", "135"]:
    dataset_name = location + prefix + i + ".asc"
    data = np.loadtxt(dataset_name, skiprows=32, delimiter=",")
    f_raw = data[:, 0]
    a = np.where(f_raw > 955)[0][0]
    b = np.where(f_raw < 1000)[0][-1]
    f = f_raw[a:b]
    s = data[a:b, 1]
    f, s = remove_noise_peaks(f, s)
    norm_counts = s[np.argmin(abs(f - 960.1))]
    s /= norm_counts
    plt.plot(f, s, label=str(int(i)) + "°")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized counts")
plt.savefig(location + "normalized.pdf", bbox_inches="tight")
plt.close()
plt.figure()

for i in ["000", "045", "090", "135"]:
    dataset_name = location + prefix + i + ".asc"
    data = np.loadtxt(dataset_name, skiprows=32, delimiter=",")
    f_raw = data[:, 0]
    a = np.where(f_raw > 955)[0][0]
    b = np.where(f_raw < 1000)[0][-1]
    f = f_raw[a:b]
    s = data[a:b, 1]
    f, s = remove_noise_peaks(f, s)
    plt.plot(f, s, label=str(int(i)) + "°")
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Counts")
plt.savefig(location + "unnormalized.pdf", bbox_inches="tight")
plt.close()
