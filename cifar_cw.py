import os
import math
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import flags

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

from cleverhans.attacks_tf import fgsm
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

import pdb

FLAGS = flags.FLAGS

def data_cifar10():
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

def main(argv=None):
    tf.set_random_seed(1234)
    sess = tf.Session()
    keras.backend.set_session(sess)

    X_train, Y_train, X_test, Y_test = data_cifar10()
    Y_train = Y_train.clip(.1 / 9., 1. - .1)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model = cnn_model(img_rows=32, img_cols=32, channels=3)
    predictions = model(x)

    def evaluate():
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    train_params = {
        'nb_epochs': FLAGS.nb_epochs, 
        'batch_size': FLAGS.batch_size, 
        'learning_rate': FLAGS.learning_rate, 
        'train_dir': FLAGS.train_dir, 
        'filename': FLAGS.filename
    }

    model_path=os.path.join(FLAGS.train_dir, FLAGS.filename)
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate, args=train_params, save=True)

    wrap = KerasModelWrapper(model)

    nb_classes = 10
    targeted = False
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'

    cw = CarliniWagnerL2(model, back='tf', sess=sess)
    adv_inputs = X_test[:10]
    adv_ys = None
    yname = "y"

    cw_params = {
        'binary_search_steps': 1,
        yname: adv_ys,
        'max_iterations': 100,
        'learning_rate': 0.1,
        'batch_size': 10,
        'initial_const': 10,
    }
    
    adv = cw.generate_np(adv_inputs, **cw_params)

    sigma = 16.0/255
    gamma = 0.00061 * 255 * 255
    alpha = 0.00061 * 255 * 255

    n_clusters = 10
    n_samples = 50

    noise = np.random.normal(0.0, sigma, adv.shape)
    adv_gauss = adv + noise

    i1 = np.repeat(np.arange(0,10), n_samples)
    i2 = np.random.randint(32, size = 10 * n_samples)
    i3 = np.random.randint(32, size = 10 * n_samples)

    sample = adv[i1, i2, i3]
    noise = np.random.normal(0.0, sigma, sample.shape)
    noisy_samples = sample + noise
    noisy_samples = np.reshape(noisy_samples, (10, n_samples, 3))

    noise = np.random.normal(0.0, sigma, adv.shape)

    adv_rdesc = np.zeros(adv.shape)
    adv_rmix = np.zeros(adv.shape)

    for img_no, img_samples in enumerate(noisy_samples):

        clusters = np.zeros((n_clusters, 3))
        clusters[0] = img_samples[0]
        
        for c_j in range(1, n_clusters):

            prob_cj = np.zeros(n_samples)

            for pix_no, pix in enumerate(img_samples):
            
                l2_min = 100000
                for c_l in range(0, c_j):
                    l2_norm_sq = np.inner(pix - clusters[c_l], pix - clusters[c_l])
                    if l2_norm_sq < l2_min:
                        l2_min = l2_norm_sq
                
                prob_cj[pix_no] = math.exp(gamma * l2_min)

            prob_cj /= prob_cj.sum()
            clusters[c_j] = img_samples[np.random.choice(n_samples, 1, p=prob_cj)]

        for pix_i in range(0, 32):
            for pix_j in range(0,32):
                c_dist_min = 100000
                c_min = np.zeros(3)
                c_sum = np.zeros(3)
                weight_sum = 0
                for c_j in clusters:
                    c_dist = np.linalg.norm(adv_gauss[img_no][pix_i][pix_j] - c_j)
                    weight_j = math.exp(-1 * alpha * c_dist * c_dist)
                    weight_sum = weight_sum + weight_j
                    c_sum = c_sum + weight_j * c_j
                    if c_dist < c_dist_min:
                        c_dist_min = c_dist
                        c_min = c_j

                adv_rdesc[img_no][pix_i][pix_j] = c_min
                adv_rmix[img_no][pix_i][pix_j] = c_sum / weight_sum

    eval_params = {'batch_size': np.minimum(nb_classes, 10)}

    adv_accuracy = 1 - model_eval(sess, x, y, predictions, adv, Y_test[:10], args=eval_params)

    print('Avg. rate of successful adv. examples without noise {0:.4f}'.format(adv_accuracy))

    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations without noise {0:.4f}'.format(percent_perturbed))

    adv_accuracy = 1 - model_eval(sess, x, y, predictions, adv_gauss, Y_test[:10], args=eval_params)

    print('Avg. rate of successful adv. examples with Gaussian noise {0:.4f}'.format(adv_accuracy))

    percent_perturbed = np.mean(np.sum((adv_gauss - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations with Gaussian noise {0:.4f}'.format(percent_perturbed))

    adv_accuracy = 1 - model_eval(sess, x, y, predictions, adv_rdesc, Y_test[:10], args=eval_params)

    print('Avg. rate of successful adv. examples with random descent {0:.4f}'.format(adv_accuracy))

    percent_perturbed = np.mean(np.sum((adv_rdesc - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations with random descent {0:.4f}'.format(percent_perturbed))
    
    adv_accuracy = 1 - model_eval(sess, x, y, predictions, adv_rmix, Y_test[:10], args=eval_params)

    print('Avg. rate of successful adv. examples with random mixture {0:.4f}'.format(adv_accuracy))

    percent_perturbed = np.mean(np.sum((adv_rmix - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations with random mixture {0:.4f}'.format(percent_perturbed))

    sess.close()

if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'models', 'Directory storing the saved model.')
    flags.DEFINE_string('filename', 'cifar10', 'Filename to save model under.')
    tf.app.run()
