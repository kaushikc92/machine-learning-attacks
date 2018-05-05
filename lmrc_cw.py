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

from PIL import Image
import scipy.misc


FLAGS = flags.FLAGS

def data_lmrc():
    img_rows = 128
    img_cols = 128
    nb_classes = 10

    X_train = np.ndarray(shape=(25000, img_rows, img_cols, 3))
    y_train = np.ndarray(shape=(25000))
    X_test = np.ndarray(shape=(5000, img_rows, img_cols, 3))
    y_test = np.ndarray(shape=(5000))

    data_dir = "training-data-mini"
    labels = os.listdir(data_dir)
    
    for i,label in enumerate(labels):
        class_path = os.path.join(data_dir, label)
        for j,filename in enumerate(os.listdir(class_path)):
            img = Image.open(os.path.join(class_path, filename))
            img_data = np.array(img)
            if j < 2500:
                y_train[2500 * i + j] = i
                X_train[2500 * i + j] = img_data
            else:
                y_test[500 * i + j - 2500] = i
                X_test[500 * i + j - 2500] = img_data
    
    rng_state = np.random.get_state()
    np.random.shuffle(y_train)
    np.random.set_state(rng_state)
    np.random.shuffle(X_train)

    rng_state = np.random.get_state()
    np.random.shuffle(y_test)
    np.random.set_state(rng_state)
    np.random.shuffle(X_test)

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

    X_train, Y_train, X_test, Y_test = data_lmrc()
    Y_train = Y_train.clip(.1 / 9., 1. - .1)

    x = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model = cnn_model(img_rows=128, img_cols=128, channels=3)
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
    adv_inputs = X_test[:100]
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

    eval_params = {'batch_size': np.minimum(nb_classes, 10)}
    original_accuracy = model_eval(sess, x, y, predictions, adv_inputs, Y_test[:100], args=eval_params)
    print('Accuracy on original images {0:.4f}'.format(original_accuracy))
    adv_accuracy = model_eval(sess, x, y, predictions, adv, Y_test[:100], args=eval_params)
    print('Accuracy on adversarial images {0:.4f}'.format(adv_accuracy))
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2, axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations without noise {0:.4f}'.format(percent_perturbed))

    for i in range(0,100):
        img = adv_inputs[i]
        filename = os.path.join("original-test-images", "img" + str(i) + ".jpg")
        scipy.misc.imsave(filename, img)

        img = adv[i]
        filename = os.path.join("adversarial-images", "img" + str(i) + ".jpg")
        scipy.misc.imsave(filename, img)

    sess.close()

if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'models', 'Directory storing the saved model.')
    flags.DEFINE_string('filename', 'lmrc', 'Filename to save model under.')
    tf.app.run()
