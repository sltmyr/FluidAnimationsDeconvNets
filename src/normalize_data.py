import numpy as np
import os
from scipy import misc
from routines import to_image_form

# TODO: time, vary x position, look at proposal .txt file. Autoencoder and recurrent neural network?

if __name__ == "__main__":
    files = []
    for yy in range(8,24):
        for xx in range(10,41):
            for rr in range(2,7):
                files.append("vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy")

    # calculate mean and stddev
    data = np.array([np.load("../res/karman_data_new/{}".format(f)) for f in files])
    print(data.shape)
    mean = np.mean(data, axis=0)
    np.save("../res/karman_data_new_norm/mean", mean)

    scale_factors = []
    for file in files:
        path = os.path.join("../res/karman_data_new", file)
        data = np.load(path)
        m = np.max(np.abs(data))
        data -= mean
        data /= m
        np.save("../res/karman_data_new_norm/{}".format(file), data)
        scale_factors.append(m)
        print("File {} factor {}".format(file, m))
    print(scale_factors)
    np.save("../res/karman_data_new_norm/scale_factors", scale_factors)
        np.save("../res/timestep_norm/{}".format(file), data)
        scale_factors.append(m)
        print("File {} factor {}".format(file, m))
    print(scale_factors)
    np.save("../res/timestep_norm/scale_factors", scale_factors)

