import numpy as np
import os.path
import matplotlib.pyplot as plt

err = np.zeros((24, 41, 9))

cnt = 0
for yy in range(8,24):
    for xx in range(10,41):
        for rr in range(4,9):
            if os.path.isfile("../res/network_output/vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy"): 
                output = np.load("../res/network_output/vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy")
                target = np.load("../res/karman_data_1711/vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy") 
                err[yy, xx, rr] = np.sqrt( np.sum((target-output)**2) / np.sum(target**2) )

err_y = np.sum(err, (1,2)) / np.maximum(1, np.count_nonzero(err, (1,2)))
err_x = np.sum(err, (0,2)) / np.maximum(1, np.count_nonzero(err, (0,2)))
err_r = np.sum(err, (0,1)) / np.maximum(1, np.count_nonzero(err, (0,1)))

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=True)
ax1.plot(range(8,24), err_y[8:24])
ax1.scatter(range(8,24), err_y[8:24])
ax1.set_xlim(7.5, 23.5)
ax1.set_ylabel("Mean $L_2$ error", usetex=True)
ax1.set_xlabel("$Y$ position")
plt.subplots_adjust(hspace = 0.5)

ax2.plot(range(10,41), err_x[10:41])
ax2.scatter(range(10,41), err_x[10:41])
ax2.set_xlim(9.5, 40.5)
ax2.set_ylabel("Mean $L_2$ error")
ax2.set_xlabel("$X$ position")

ax3.plot(range(4,9), err_r[4:9])
ax3.scatter(range(4,9), err_r[4:9])
ax3.set_xlim(3.5, 8.5)
ax3.set_ylabel("Mean $L_2$ error")
ax3.set_xlabel("Radius")


plt.show()