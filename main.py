"""
Yelp Restaurant Photo Classification Challenge
"""
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from extract import create_graph, iterate_mini_batches, batch_pool3_features
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gzip,cPickle
import os

number_of_folder = 100

def load_pool3_data():                                                  """ load pre-computed CNN codes  """
    """ Update these file names after you serialize pool_3 values """

    X_test_file = 'conc_CNN_code.npy'
    y_test_file = 'conc_CNN_label.npy'
    X_train_file = 'conc_CNN_code.npy'
    y_train_file = 'conc_CNN_label.npy'
    return np.load(X_train_file), np.load(y_train_file), np.load(X_test_file), np.load(y_test_file)

classes = np.array(['not-good','good'])                             """ name of the classes """

X_train_pool, y_train_pool, X_test_pool, y_test_pool = load_pool3_data()

X_train_pool3 = X_train_pool[31250:231251,:]                            """creating the training data and testing data for a specifc label"""
y_train_pool3 = y_train_pool[31250:231251,7]                            """ change the label here 0,1,2,3,4,5,6,7,8 """    
X_test_pool3 = X_train_pool[0:31250,:]
y_test_pool3 = y_train_pool[0:31250,7]                                  """ change the label here 0,1,2,3,4,5,6,7,8 """

print(X_train_pool3.shape)
y_train_pool3 = np.transpose(y_train_pool3)
y_test_pool3 = np.transpose(y_test_pool3)
print(y_train_pool3.shape)
X_train, X_validation, Y_train, y_validation = cross_validation.train_test_split(X_train_pool3, y_train_pool3, test_size=0.30, random_state=40)


print 'Training data shape: ', X_train_pool3.shape
print 'Training labels shape: ', y_train_pool3.shape
print 'Test data shape: ', X_test_pool3.shape
print 'Test labels shape: ', y_test_pool3.shape

#
# Tensorflow stuff
# #

FLAGS = tf.app.flags.FLAGS
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'
BOTTLENECK_TENSOR_SIZE = 2048
tf.app.flags.DEFINE_integer('how_many_training_steps', 100,
                            """How many training steps to run before ending.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          """How large a learning rate to use when training.""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")
tf.app.flags.DEFINE_integer('eval_step_interval', 10,
                            """How often to evaluate the training results.""")



# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def ensure_name_has_port(tensor_name):
    """Makes sure that there's a port number at the end of the tensor name.
    Args:
      tensor_name: A string representing the name of a tensor in a graph.
    Returns:
      The input string with a :0 appended if no port was specified.
    """
    if ':' not in tensor_name:
        name_with_port = tensor_name + ':0'
    else:
        name_with_port = tensor_name
    return name_with_port

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_final_training_ops(graph, class_count, final_tensor_name,
                           ground_truth_tensor_name):
    """Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    The set up for the softmax and fully-connected layers is based on:
    https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args:
      graph: Container for the existing model's Graph.
      class_count: Integer of how many categories of things we're trying to
      recognize.
      final_tensor_name: Name string for the new final node that produces results.
      ground_truth_tensor_name: Name string of the node we feed ground truth data
      into.
    Returns:
      Nothing.
    """
    bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        BOTTLENECK_TENSOR_NAME))
    layer_weights = tf.Variable(
        tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
        name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    logits = tf.matmul(bottleneck_tensor, layer_weights,
                       name='final_matmul') + layer_biases
    tf.nn.softmax(logits, name=final_tensor_name)
    ground_truth_placeholder = tf.placeholder(tf.float32,
                                              [None, class_count],
                                              name=ground_truth_tensor_name)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, ground_truth_placeholder)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy_mean)
    return train_step, cross_entropy_mean

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def add_evaluation_step(graph, final_tensor_name, ground_truth_tensor_name):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
      graph: Container for the existing model's Graph.
      final_tensor_name: Name string for the new final node that produces results.
      ground_truth_tensor_name: Name string for the node we feed ground truth data
      into.
    Returns:
      Nothing.
    """
    result_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        final_tensor_name))
    ground_truth_tensor = graph.get_tensor_by_name(ensure_name_has_port(
        ground_truth_tensor_name))
    correct_prediction = tf.equal(
        tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return evaluation_step

def encode_one_hot(nclasses,y):
    return np.eye(nclasses)[y]

def do_train(sess,X_input, Y_input, X_validation, Y_validation):
    ground_truth_tensor_name = 'ground_truth'
    mini_batch_size = 256
    n_train = X_input.shape[0]

    graph = create_graph()

    train_step, cross_entropy = add_final_training_ops(
        graph, len(classes), FLAGS.final_tensor_name,
        ground_truth_tensor_name)

    init = tf.initialize_all_variables()
    sess.run(init)

    evaluation_step = add_evaluation_step(graph, FLAGS.final_tensor_name, ground_truth_tensor_name)

    # Get some layers we'll need to access during training.
    bottleneck_tensor = graph.get_tensor_by_name(ensure_name_has_port(BOTTLENECK_TENSOR_NAME))
    ground_truth_tensor = graph.get_tensor_by_name(ensure_name_has_port(ground_truth_tensor_name))

    i=0
    epocs = 20
    validation_acc_vector = []
    training_acc_vector  = []
    cross_entropy_vector = []
    for epoch in xrange(epocs):
        shuffledRange = np.random.permutation(n_train)
        y_one_hot_train = encode_one_hot(len(classes), Y_input)
        y_one_hot_validation = encode_one_hot(len(classes), Y_validation)
        shuffledX = X_input[shuffledRange,:]
        shuffledY = y_one_hot_train[shuffledRange]
        for Xi, Yi in iterate_mini_batches(shuffledX, shuffledY, mini_batch_size):
            sess.run(train_step,
                     feed_dict={bottleneck_tensor: Xi,
                                ground_truth_tensor: Yi})
            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                  [evaluation_step, cross_entropy],
                  feed_dict={bottleneck_tensor: Xi,
                             ground_truth_tensor: Yi})
                validation_accuracy = sess.run(
                  evaluation_step,
                  feed_dict={bottleneck_tensor: X_validation,
                             ground_truth_tensor: y_one_hot_validation})
                print('%s: Step %d: Train accuracy = %.1f%%, Cross entropy = %f, Validation accuracy = %.1f%%' %
                    (datetime.now(), i, train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))
                cross_entropy_vector.append(cross_entropy_value)
                training_acc_vector.append(train_accuracy * 100)
                validation_acc_vector.append(validation_accuracy * 100)
            i+=1
    print("cross entropy vector length is "+str(len(cross_entropy_vector)))
    x_ax = np.arange(0,len(cross_entropy_vector))                           """ plotting the training accuarcy vs iterations"""
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(x_ax, training_acc_vector)
    plt.xlabel('Iterations')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs Number of Iterations')
    fig.savefig('training_acc.jpg')
    plt.close(fig)
    
    x_ax = np.arange(0,len(cross_entropy_vector))                       """ plotting the validation accuarcy vs iterations"""
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(x_ax, validation_acc_vector)
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Number of Iterations')
    fig.savefig('validation_acc.jpg')
    plt.close(fig)
 
    x_ax = np.arange(0,len(cross_entropy_vector))                       """ plotting the cross-entropy error vs iterations"""
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(x_ax, cross_entropy_vector)
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy value')
    plt.title('Cross Entropy Value vs Number of Iterations')
    fig.savefig('cross_entropy_value.jpg')
    plt.close(fig)

    test_accuracy = sess.run(                                       """ calculating the test_set accuracy"""    
        evaluation_step,
        feed_dict={bottleneck_tensor: X_test_pool3,
                   ground_truth_tensor: encode_one_hot(len(classes), y_test_pool3)})
    print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

def show_test_images(sess, X_features, Y):
    n = X_features.shape[0]

    def rand_ordering():
        return np.random.permutation(n)

    def sequential_ordering():
        return range(n)
    truth_class=[]
    predicted_class = []
    for i in sequential_ordering():
        Xi_features=X_features[i,:].reshape(1,2048)
        result_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(FLAGS.final_tensor_name))
        probs = sess.run(result_tensor,
                         feed_dict={'pool_3/_reshape:0': Xi_features})
        predicted_class.append(np.argmax(probs))
        truth_class.append(Y[i])
    predicted_class = np.asarray(predicted_class)                   """ saving the predicted class for F1 score """
    np.save('predicted_class.npy',predicted_class)
    truth_class=np.asarray(truth_class)
    np.save('truth_class.npy',truth_class)                          """ saving the ground truth""" 
       


sess = tf.InteractiveSession()
do_train(sess,X_train,Y_train,X_validation,y_validation)
show_test_images(sess,X_test_pool3, y_test_pool3)
