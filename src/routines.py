import tensorflow as tf
import numpy as np
import csv
import math
import matplotlib.pyplot as plt


def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def leaky_relu(x, alpha):
    return tf.maximum(x, alpha * x)


def to_image_form(data):
    return np.reshape(data, (32, 64, 2))


def load_data():
    training_image = []
    training_val = []
    for i in range(1, 32):
        if i == 16: continue
        path = "../res/karman_data_norm/vel" + str(i) + ".npy"
        training_image.append(np.load(path).flatten())
        training_val.append(i / 32.0)

    training_data = [np.reshape(training_val, (30, 1)), training_image]
    return training_data

def shuffle(data):
	#np.random.seed(239)
	shuffle_idx = np.random.permutation(len(data[0]))
	data[0][:] = data[0][shuffle_idx]
	data[1][:] = data[1][shuffle_idx]

def load_data_3():
    image = []
    vec = []

    for yy in range(8,24):
    	for xx in range(10,41):
    		for rr in range(2,7):
        		path = "../res/karman_data_new/vel_"+str(xx)+"_"+str(yy)+"_"+str(rr)+".npy"
        		image.append(np.load(path).flatten())
        		vec.append([xx , yy , rr])

    data = [np.asarray(vec), np.asarray(image)]
    shuffle(data)
    return data

def get_scale_factor(x_pos, y_pos, r):
    index = int((y_pos-8)*155 + (x_pos-10)*5 + r-2)
    return np.load("../res/karman_data_new_norm/scale_factors.npy")[index]


def get_time_scale_factor(x):
    index = int(x[0] * 32 - 1) * 50 + x[1]
    return np.load("../res/timestep_norm/scale_factors.npy")[index]


def save_csv(data, path):
    with open(path, "w") as file:
        writer = csv.writer(file)
        for k, v in data.items():
            writer.writerow([k, v])


def plot(img,loc):
	name = str(loc[0]) + "_" + str(loc[1]) + "_" + str(loc[2])
	#name = str(loc[1])
	real_flow = np.load("../res/karman_data_new/vel_"+name+".npy")
	net_flow = img

	# takes ONE real flow and ONE output from network and compares them
	real_flow = real_flow.transpose((1, 0, 2))
	net_flow = net_flow.transpose((1, 0, 2))
	image_size = real_flow.shape

	skip = 2
	X, Y = np.mgrid[0:image_size[0]:skip, 0:image_size[1]:skip]
	[f, (ax1, ax2, ax3)] = plt.subplots(3, sharex=True, sharey=True)
	ax1.quiver(X, Y, real_flow[::skip, ::skip, 0], real_flow[::skip, ::skip, 1], units='inches')
	circle1 = plt.Circle((loc[0], loc[1]), loc[2], color='r')
	circle2 = plt.Circle((loc[0], loc[1]), loc[2], color='r')
	circle3 = plt.Circle((loc[0], loc[1]), loc[2], color='r')
	ax1.add_artist(circle1)

	ax1.set_title("Real flow")
	ax1.set_xlim(0, image_size[0])
	ax1.set_ylim(0, image_size[1])
	ax2.set_title("Output of network")
	ax2.quiver(X, Y, net_flow[::skip, ::skip, 0], net_flow[::skip, ::skip, 1], units='inches')
	ax2.add_artist(circle2)
	# compute error
	diff_flow = (real_flow[:, :, 0] - net_flow[:, :, 0]) ** 2 + (real_flow[:, :, 1] - net_flow[:, :, 1]) ** 2
	diff_norm = math.sqrt(np.sum(diff_flow))
	
	real_flow_sq = real_flow[:, :, 0] ** 2 + real_flow[:, :, 1] ** 2
	real_norm = math.sqrt(np.sum(real_flow_sq))
	print("Average error: %f" % (diff_norm / real_norm))
	
	real_max = np.amax(real_flow_sq)
	diff_max = np.amax(diff_flow)

	ax3.set_title("Plot of velocity differences (real-net)")
	ax3.quiver(X, Y, real_flow[::skip, ::skip, 0] - net_flow[::skip, ::skip, 0],
	           real_flow[::skip, ::skip, 1] - net_flow[::skip, ::skip, 1], scale=real_max, units='inches')
	ax3.add_artist(circle3)
	plt.show()
	exit()