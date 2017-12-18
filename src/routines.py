import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm 


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def denormalize(output_image):
    output_image[:, :, 0] *= np.load("../res/karman_data_1711_norm/norm_factor_x.npy")
    output_image[:, :, 1] *= np.load("../res/karman_data_1711_norm/norm_factor_y.npy")
    output_image[:, :, 0] += np.load("../res/karman_data_1711_norm/mean_x.npy")
    output_image[:, :, 1] += np.load("../res/karman_data_1711_norm/mean_y.npy")
    return output_image


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)

def to_image_form(data):
    return np.reshape(data, (32, 64, 2))

def shuffle(data):
	np.random.seed(239)
	shuffle_idx = np.random.permutation(len(data[0]))
	data[0][:] = data[0][shuffle_idx]
	data[1][:] = data[1][shuffle_idx]

def load_data_3():
    image = []
    vec = []

    for yy in range(8,24):
    	for xx in range(10,41):
    		for rr in range(4,9):
        		path = "../res/karman_data_1711_norm/vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy"
        		image.append(np.load(path).flatten())
        		vec.append([xx , yy , rr])

    data = [np.asarray(vec), np.asarray(image)]
    shuffle(data)
    return data

def save_csv(data, path):
    with open(path, "w") as file:
        writer = csv.writer(file)
        for k, v in data.items():
            writer.writerow([k, v])


def plot(img,loc):
	name = str(loc[0]) + "_" + str(loc[1]) + "_" + str(loc[2])
	real_flow = np.load("../res/karman_data_1711/vel_"+name+".npy")
	net_flow = img

	# takes ONE real flow and ONE output from network and compares them
	real_flow = real_flow.transpose((1, 0, 2))
	net_flow = net_flow.transpose((1, 0, 2))
	image_size = real_flow.shape

	skip = 3
	X, Y = np.mgrid[1:image_size[0]:skip, 1:image_size[1]:skip]
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
	
	# velocity plot of the real flow (ground truth)
	
	widths = 1.0
	ax1.quiver(X, Y, real_flow[1::skip, 1::skip, 0], real_flow[1::skip, 1::skip, 1], scale = 5.0, units='inches', linewidths=widths)
	circle1 = plt.Circle((loc[0], loc[1]), loc[2], color='k')
	circle2 = plt.Circle((loc[0], loc[1]), loc[2], color='k')
	circle3 = plt.Circle((loc[0], loc[1]), loc[2], color='k')
	circle4 = plt.Circle((loc[0], loc[1]), loc[2], color='k')
	ax1.add_artist(circle1)

	ax1.set_title("Real flow")
	ax1.set_xlim(0, image_size[0]-1)
	ax1.set_ylim(0, image_size[1]-1)
	
	# velocity plot of the generated flow (network output)
	ax2.set_title("Network output")
	ax2.quiver(X, Y, net_flow[1::skip, 1::skip, 0], net_flow[1::skip, 1::skip, 1], scale = 5.0, units='inches', linewidths=widths)
	ax2.add_artist(circle2)
	
	# error computations
	diff_flow = (real_flow[:, :, 0] - net_flow[:, :, 0]) ** 2 + (real_flow[:, :, 1] - net_flow[:, :, 1]) ** 2
	diff_norm = math.sqrt(np.sum(diff_flow))
	
	real_flow_sq = real_flow[:, :, 0] ** 2 + real_flow[:, :, 1] ** 2
	real_norm = math.sqrt(np.sum(real_flow_sq))
	print("Average error: %f" % (diff_norm / real_norm))
	
	# velocity difference
	real_max = np.amax(real_flow_sq)
	diff_max = np.amax(diff_flow)
	ax3.set_title("Difference")
	ax3.quiver(X, Y, real_flow[1::skip, 1::skip, 0] - net_flow[1::skip, 1::skip, 0],
	           real_flow[1::skip, 1::skip, 1] - net_flow[1::skip, 1::skip, 1], scale = 5.0, units='inches', linewidths=widths)
	ax3.add_artist(circle3)
	
	# velocity difference (color-coded)
	ax4.set_title("Difference (color-coded)")
	Xall, Yall = np.mgrid[0:image_size[0], 0:image_size[1]]
	ax4.pcolormesh(Xall, Yall, diff_flow, cmap = cm.Reds)
	ax4.add_artist(circle4)
	
	plt.show()