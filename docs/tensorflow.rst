Using Tensorflow with Gradientzoo
=================================

This doc will demonstrate how to save and load your trained neural network in
Python using Tensorflow.


Initialization
--------------

.. code:: python

    from gradientzoo import TensorflowGradientzoo

This is the client that you'll create to save and load your model to the
service.

:param username: The gradientzoo username of the project you want to connect to
:param model_slug: The slug (short url-safe name) of the model
:param api_base: *Optional* URL prefix to specify where the gradientzoo API is located, which is especially useful if you are running your own instance of gradientzoo
:param auth_token_id: Your authentication token as provided by the service
:param default_dir: *Optional* When a download directory is asked for later, this is what it will default to if you pass None

**Note:** For convenience and aesthetics, you can pass both the username and model_slug as ``username/model_slug`` to the username parameter.

.. code:: python

    # Examples of how to instantiate the client
    zoo = TensorflowGradientzoo('exampleuser/modelslug')
    zoo = TensorflowGradientzoo('exampleuser', 'modelslug', auth_token_id='12345af')
    zoo = TensorflowGradientzoo('exampleuser/modelslug', api_base='https://api2.gradientzoo.com')


.. rubric:: Examples

The next few examples will assume ``zoo``, an instantiated TensorflowGradientzoo
client:

.. code:: python

    from gradientzoo import TensorflowGradientzoo
    zoo = TensorflowGradientzoo('exampleuser/modelslug')


Loading a Model
---------------

If you want to load a model from the remote cloud, that can be done by calling
the ``load`` method on the client.

:param session: The Tensorflow session you want to load the weights into
:param filename: *Optional* The filename of the thing you would like to download
:param id: *Optional* If there is a specific version of the file you want to download, provide the file id (as seen on the website) here
:param dir: *Optional* The directory to download the file to

:returns file_model: A dictionary with metadata about the file you downloaded

.. code:: python

    # Example of how to load a Tensorflow session
    with tf.Session() as sess:
        # Load latest weights from Gradientzoo
        zoo.load(sess)


Saving a Model
--------------

If you want to save your moel to the remote cloud, simply call the
``upload_file`` method on the client.

:param session: The Tensorflow session you want to save the weights for
:param filename: *Optional* The name of the file you're uploading
:param metadata: *Optional* A bag of key/value pairs to be associated with this file (must be JSON encodable)
:param dir: *Optional* The temporary directory to save the model weights to before uploading them

.. code:: python

    # Example of how to save a Tensorflow session
    zoo.save(sess)
