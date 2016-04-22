Using Keras with Gradientzoo
============================

This doc will demonstrate how to save and load your trained neural network in
Python using Keras.


Initialization
--------------

.. code:: python

    from gradientzoo import KerasGradientzoo

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
    zoo = KerasGradientzoo('exampleuser/modelslug')
    zoo = KerasGradientzoo('exampleuser', 'modelslug', auth_token_id='12345af')
    zoo = KerasGradientzoo('exampleuser/modelslug', api_base='https://api2.gradientzoo.com')


.. rubric:: Examples

The next few examples will assume ``zoo``, an instantiated KerasGradientzoo
client:

.. code:: python

    from gradientzoo import KerasGradientzoo
    zoo = KerasGradientzoo('exampleuser/modelslug')


Loading a Model
---------------

If you want to load a model from the remote cloud, that can be done by calling
the ``load`` method on the client.

:param model: The Keras model you want to load the weights into
:param filename: *Optional* The filename of the thing you would like to download
:param id: *Optional* If there is a specific version of the file you want to download, provide the file id (as seen on the website) here
:param dir: *Optional* The directory to download the file to

:returns file_model: A dictionary with metadata about the file you downloaded

.. code:: python

    # Example of how to load a model
    zoo.load(your_keras_model)


Saving a Model
--------------

If you want to save your moel to the remote cloud, simply call the
``upload_file`` method on the client.

:param model: The Keras model you want to save the weights for
:param filename: *Optional* The name of the file you're uploading
:param metadata: *Optional* A bag of key/value pairs to be associated with this file (must be JSON encodable)
:param dir: *Optional* The temporary directory to save the model weights to before uploading them

.. code:: python

    # Example of how to save a model
    zoo.save(your_keras_model)


Creating a Keras callback that will save automatically
------------------------------------------------------

Often times you don't want to manage saving the model yourself. To make this
easier, we provide a Keras callback that will save every N batches or every N
epochs.  To get a callback, call the ``make_callback`` method on the client.

:param model: The Keras model you want to save the weights for
:param after_batches: *Optional* The number of batches after which to save
:param after_epochs: *Optional* The number of epochs after which to save (defaults to 1)

.. code:: python

  # Example of how to automatically save a model using a callback
  zoo_callback = zoo.make_callback(your_model)
  your_model.fit(X_train, Y_train, # ...
                 callbacks=[zoo_callback])
