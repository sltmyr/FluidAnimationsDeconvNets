from routines import *

batch_size = 50
epochs = 1

def conv_layer(input, width, height, channels, linear=False):
    output_shape = [batch_size, width, height, channels]
    strides = [1, 2, 2, 1]
    weights = weight_variable([5, 5, output_shape[-1], int(input.get_shape()[-1])])
    output = tf.nn.conv2d_transpose(input,
                                    weights,
                                    output_shape=output_shape,
                                    strides=strides)
    #leaky ReLU
    if not linear:
        output = tf.maximum(output, 0.2*output)
    return output

def fc_layer(input, input_size, output_size):
    weights = weight_variable([input_size, output_size])
    biases = bias_variable([1, output_size])
    return tf.tanh(tf.add(tf.matmul(input, weights), biases))


def create_net(x):
    # fully connected layers
    x = fc_layer(x, 3, 16)
    x = fc_layer(x, 16, 256)
    x = fc_layer(x, 256, 512)
    x = tf.layers.batch_normalization(x, 1)
    x = tf.reshape(x, [batch_size, 4, 2, 64])

    # deconvolutional layers
    x = conv_layer(x, 8, 4, 32)
    x = conv_layer(x, 16, 8, 16)
    x = conv_layer(x, 32, 16, 8)
    x = conv_layer(x, 64, 32, 2, linear=True)

    output = tf.reshape(x, [batch_size, 4096])
    return tf.tanh(output)
    
def compute_l2_loss(output, ground_truth):
    return tf.reduce_mean(
             		tf.sqrt(
                    	tf.reduce_sum(tf.pow(tf.subtract(ground_truth,output), 2),reduction_indices=[1])
                       	/ tf.reduce_sum(tf.pow(ground_truth,2),reduction_indices=[1])))

def create_trainer(output, ground_truth):
    loss = compute_l2_loss(output, ground_truth)
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.01, global_step, 50, 0.95)
    #lr = tf.train.piecewise_constant(global_step, [5000, 8000], [0.1, 0.05, 0.01])
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return train_step, loss

def validate(x, output, sess, validation_data):
    error = 0.
    num_batches = int(len(validation_data[0])/batch_size)
    for batch in range(num_batches):
        validation_batch_x = validation_data[0][batch*batch_size:(batch+1)*batch_size]
        validation_batch_y = validation_data[1][batch*batch_size:(batch+1)*batch_size]
        net_output_y = sess.run(output, feed_dict={x: validation_batch_x})
        error += compute_l2_loss(net_output_y, validation_batch_y).eval()
   
    return error/num_batches

def train_epoch(x, ground_truth, train_step, loss, sess, training_data):
    shuffle(training_data)
    train_error = 0.0
    num_batches = int(len(training_data[0])/batch_size)
    for batch in range(num_batches):
        training_batch_x = training_data[0][batch*batch_size:(batch+1)*batch_size]
        training_batch_y = training_data[1][batch*batch_size:(batch+1)*batch_size]
        _, loss_val = sess.run([train_step, loss], feed_dict={x: training_batch_x, ground_truth: training_batch_y})
        train_error += loss_val
    return train_error/num_batches

def test(x, output, sess, test_data):
    test_error = validate(x, output, sess, test_data)
    net_output_y = sess.run(output, feed_dict={x: test_data[0][0:batch_size]})

    output_image = to_image_form(net_output_y[0])
    output_image[:, :, 0] *= np.load("../res/karman_data_1711_norm/norm_factor_x.npy")
    output_image[:, :, 1] *= np.load("../res/karman_data_1711_norm/norm_factor_y.npy")
    output_image[:, :, 0] += np.load("../res/karman_data_1711_norm/mean_x.npy")
    output_image[:, :, 1] += np.load("../res/karman_data_1711_norm/mean_y.npy")
    
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
    test_data = [data[0][1800:2100,:], data[1][1800:2100,:]]
    validation_data = [data[0][2100:2400,:], data[1][2100:2400,:]]

    loss_data = {}
    val_error_data = {}
    for i in range(epochs):
        loss_data[i] = train_epoch(x, ground_truth, train_step, loss, sess, training_data)
        print("Epoch: ", i)
        print("Loss: ", loss_data[i])
        val_error_data[i] = validate(x, output, sess, validation_data)

    save_csv(loss_data, "../res/train_loss.csv")
    save_csv(val_error_data, "../res/validation_error.csv")

    test_error = test(x, output, sess, test_data)
    print("Mean Squared Error on test set: ", test_error)

if __name__ == "__main__":
    run()
