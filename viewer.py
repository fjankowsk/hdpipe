# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

dtype = [("snr","float"), ("samp_nr","int"), ("time","float"), ("filter","int"),
        ("dmtrial","int"), ("dm","float"),
        ("cluster_nr","int"), ("start","int"), ("end","int")]

data = np.genfromtxt("all.txt", dtype=dtype)

# remove all low-snr candidates and the ones that are really wide
mask = (data["snr"] > 9.0) & (data["filter"] <= 4)
data = data[mask]

fig = plt.figure()
ax = fig.add_subplot(111)

sc = ax.scatter(data["dm"] + 1, data["snr"],
c=data["filter"],
marker="o")
plt.colorbar(sc, label="Filter number")

ax.set_xscale("log")
ax.grid()
ax.set_xlabel("DM+1 [pc/cm3]")
ax.set_ylabel("S/N")

plt.show()
