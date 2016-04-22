# Gradientzoo Python bindings

[![Documentation Status](https://readthedocs.org/projects/python-gradientzoo/badge/?version=latest)](http://python-gradientzoo.readthedocs.org/en/latest/?badge=latest)

This is a Python library for Gradientzoo's API - Version and share your trained
neural network models.  Loading a pre-trained neural network is easy with
Gradientzoo. Here's how easy it is to load a model with Tensorflow (full
example below):

```python
import tensorflow as tf
from gradientzoo import TensorflowGradientzoo

# (build MNIST graph here)

with tf.Session() as sess:
    # Load latest weights from Gradientzoo
    TensorflowGradientzoo('ericflo/mnist').load(sess)

    # Graph is now ready to use!
```

## Features

Supports saving models in [Keras](http://keras.io/), variables in [Tensorflow](https://www.tensorflow.org), and networks in [Lasagne](http://lasagne.readthedocs.org/en/latest/), and regular old files using Python with your framework of choice.

## Installation

You don't need this source code unless you want to modify the
package. If you just want to use the Gradientzoo Python bindings, you
should run:

    pip install --upgrade gradientzoo

or

    easy_install --upgrade gradientzoo

See http://www.pip-installer.org/en/latest/index.html for instructions
on installing pip. If you are on a system with easy_install but not
pip, you can use easy_install instead. If you're not using virtualenv,
you may have to prefix those commands with `sudo`. You can learn more
about virtualenv at http://www.virtualenv.org/

To install from source, run:

    python setup.py install


## Documentation

Please see http://python-gradientzoo.readthedocs.org/ for the most up-to-date
documentation or visit a project page to see project-specific instructions,
e.g. https://www.gradientzoo.com/ericflo/mnist

## Contribute

- Issue Tracker: https://github.com/gradientzoo/python-gradientzoo/issues
- Source Code: https://github.com/gradientzoo/python-gradientzoo

## Support

If you are having issues, please let us know at support@gradientzoo.com

## Full Tensorflow Example

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data, mnist
from gradientzoo import TensorflowGradientzoo

learning_rate = 0.01
batch_size = 100

# Build MNIST graph
images_placeholder = tf.placeholder(tf.float32,
                                    shape=(batch_size, mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
logits = mnist.inference(images_placeholder, 128, 32)
loss = mnist.loss(logits, labels_placeholder)
train_op = mnist.training(loss, learning_rate)
eval_correct = mnist.evaluation(logits, labels_placeholder)

# Start a Tensorflow session
with tf.Session() as sess:
    # Load latest weights from Gradientzoo
    TensorflowGradientzoo('ericflo/mnist').load(sess)

    # Read in some data
    data_sets = input_data.read_data_sets('data', False)

    # Test the trained network on the dataset
    true_count = 0
    for step in xrange(data_sets.test.num_examples // batch_size):
        images_feed, labels_feed = data_sets.test.next_batch(batch_size, False)

        true_count += sess.run(eval_correct, feed_dict={
            images_placeholder: images_feed,
            labels_placeholder: labels_feed,
        })

    precision = true_count / float(data_sets.test.num_examples)
    print('Num Examples: %d  Num Correct: %d  Precision: %0.04f' %
          (data_sets.test.num_examples, true_count, precision))
```