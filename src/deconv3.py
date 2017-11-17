from routines import *
import random

batch_size = 50
epochs = 100

def conv_layer(input, width, height, channels, linear=False):
    output_shape = [batch_size, width, height, channels] #[batch_size, height, width, channels]
    strides = [1, 2, 2, 1]
    weights = weight_variable([5, 5, output_shape[-1],int(input.get_shape()[-1])])
    output = tf.nn.conv2d_transpose(input,
                                     weights, 
                                     output_shape=output_shape,
                                     strides=strides)
    #leaky ReLU
    if linear==False:
        output = tf.maximum(output, 0.2*output)
    return output


def create_net(x):
    # fully connected layer
    fc_neurons = 4096
    w_fc0 = weight_variable([3,fc_neurons])
    b_fc0 = bias_variable([1,fc_neurons])
    x = tf.tanh(tf.add(tf.matmul(x,w_fc0),b_fc0))
    x = tf.reshape(x, [batch_size,8,4,128]) #hidden: 16x8x8, 8 channels of 16x8 layers

    # deconvolutional layers
    x = conv_layer(x,16,8,32)
    x = conv_layer(x,32,16,8)
    x = conv_layer(x,64,32,2, linear=True)


    # fc_neurons = 4096
    # w_fc0 = weight_variable([3,fc_neurons])
    # b_fc0 = bias_variable([1,fc_neurons])
    # x = tf.tanh(tf.add(tf.matmul(x,w_fc0),b_fc0))
    # x = tf.reshape(x, [batch_size,4,2,512]) #hidden: 16x8x8, 8 channels of 16x8 layers

    # # deconvolutional layers
    # x = conv_layer(x,8,4,256)
    # x = conv_layer(x,16,8,128)
    # x = conv_layer(x,32,16,64)
    # x = conv_layer(x,64,32,2, linear=True)

    output = tf.reshape(x, [batch_size,4096])
    return tf.tanh(output)

def create_trainer(output, ground_truth):
    loss = tf.reduce_mean(tf.reduce_sum(tf.pow(ground_truth - output, 2), reduction_indices=[1]))
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.00001,global_step,1000,0.95)
    #lr = tf.train.piecewise_constant(global_step, [5000, 8000], [0.1, 0.05, 0.01])
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step, loss

def train(x, ground_truth, train_step, loss, sess, training_data):
    loss_data = {}
    for i in range(epochs):
        shuffle(training_data)
        for batch in range(int(len(training_data[0])/batch_size)):
            training_batch_x = training_data[0][batch*batch_size:(batch+1)*batch_size]
            training_batch_y = training_data[1][batch*batch_size:(batch+1)*batch_size]
            _, loss_val = sess.run([train_step, loss], feed_dict={x: training_batch_x, ground_truth: training_batch_y})
        print("Epoch {}: Loss = {}".format(i, loss_val))
        loss_data[i] = loss_val

    save_csv(loss_data, "../res/training_memorize_all.csv")

def validate(x, output, sess, validation_data):
    error = 0.
    num_batches = int(len(validation_data[0])/batch_size)
    for batch in range(num_batches):
            validation_batch_x = validation_data[0][batch*batch_size:(batch+1)*batch_size]
            validation_batch_y = validation_data[1][batch*batch_size:(batch+1)*batch_size]
            net_output_y = sess.run(output, feed_dict={x: validation_batch_x})
            error += tf.reduce_mean(
                        tf.reduce_sum(tf.pow(validation_batch_y- net_output_y, 2),reduction_indices=[1]) 
                        / tf.reduce_sum(tf.pow(validation_batch_y,2),reduction_indices=[1])).eval()
    return error/num_batches

def test(x, output, sess, test_data):
    test_error = validate(x, output, sess, test_data)
    net_output_y = sess.run(output, feed_dict={x: test_data[0][0:batch_size]})
    
    output_image = to_image_form(net_output_y[0])
    output_image[:,:,0] *= np.load("../res/karman_data_1711_norm/norm_factor_x.py")
    output_image[:,:,1] *= np.load("../res/karman_data_1711_norm/norm_factor_y.py")
    output_image[:,:,0] += np.load("../res/karman_data_1711_norm/mean_x.py")
    output_image[:,:,1] += np.load("../res/karman_data_1711_norm/mean_y.py")
    
    np.save("../res/net_image",output_image)
    plot(output_image,test_data[0][0])
    
    return test_error

def run():
    x = tf.placeholder(tf.float32, [None, 3])
    output = create_net(x)
    ground_truth = tf.placeholder(tf.float32, [None, 4096])
    train_step, loss = create_trainer(output, ground_truth)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    data = load_data_3()
    training_data = [data[0][:1800,:], data[1][:1800,:]]
    test_data = [data[0] [1800:2100,:], data[1][1800:2100,:]]
    validation_data = [data[0][2100:2400,:], data[1][2100:2400,:]]

    train(x, ground_truth, train_step, loss, sess, training_data)

    validation_error = validate(x, output, sess, validation_data)
    test_error = test(x, output, sess, test_data)   
    print("Mean Squared Error on validation set: ",validation_error)
    print("Mean Squared Error on test set: ", test_error)

if __name__ == "__main__":
    run()