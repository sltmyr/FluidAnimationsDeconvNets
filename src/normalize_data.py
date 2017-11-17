import numpy as np
import os

if __name__ == "__main__":
    files = []
    for yy in range(8,24):
        for xx in range(10,41):
            for rr in range(4,9):
                files.append("vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy")

    # calculate mean and norm factor
    data = np.array([np.load("../res/karman_data_1711/{}".format(f)) for f in files])
    
    mean_x = np.mean(data[:,:,:,0], axis=0)
    mean_y = np.mean(data[:,:,:,1], axis=0)
    
    data[:,:,:,0] -= mean_x
    data[:,:,:,1] -= mean_y
    
    norm_factor_x = np.max(np.abs(data[:,:,:,0]))
    norm_factor_y = np.max(np.abs(data[:,:,:,1]))
    
    np.save("../res/karman_data_1711_norm/mean_x", mean_x)
    np.save("../res/karman_data_1711_norm/mean_y", mean_y)
    
    np.save("../res/karman_data_1711_norm/norm_factor_x", norm_factor_x)
    np.save("../res/karman_data_1711_norm/norm_factor_y", norm_factor_y)
    
    for file in files:
        path = os.path.join("../res/karman_data_1711", file)
        data = np.load(path)
        
        data[:,:,0] -= mean_x
        data[:,:,1] -= mean_y
        
        data[:,:,0] /= norm_factor_x
        data[:,:,1] /= norm_factor_y
        
        np.save("../res/karman_data_1711_norm/{}".format(file), data)
