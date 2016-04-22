Using plain old Python with Gradientzoo
=======================================

This doc will demonstrate how to save and load your trained neural network in
Python.  This is useful either for writing your own library integration, or for
doing something outside of what the standard integrations expect.


Initialization
--------------

.. code:: python

    from gradientzoo import Gradientzoo

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
    zoo = Gradientzoo('exampleuser/modelslug')
    zoo = Gradientzoo('exampleuser', 'modelslug', auth_token_id='12345af')
    zoo = Gradientzoo('exampleuser/modelslug', api_base='https://api2.gradientzoo.com')


.. rubric:: Examples

The next few examples will assume ``zoo``, an instantiated Gradientzoo client:

.. code:: python

    from gradientzoo import Gradientzoo
    zoo = Gradientzoo('exampleuser/modelslug')


Loading a Model
---------------

If you want to load a model from the remote cloud, that can be done by calling
the ``download_file`` method on the client.

:param filename: The filename of the thing you would like to download
:param id: *Optional* If there is a specific version of the file you want to download, provide the file id (as seen on the website) here
:param dir: *Optional* The directory to download the file to
:param chunk_size: *Optional* The size of the chunks to read and flush to disk

:returns file_model: A dictionary with metadata about the file you downloaded

.. code:: python

    # Example of how to download a file
    filepath, file_model = zoo.download_file('model.npz')

    with open(filepath, 'r') as f:
        data = f.read()

    # Now load the data into your model however your framework or library works


Saving a Model
--------------

If you want to save your moel to the remote cloud, simply call the
``upload_file`` method on the client.

:param filename: The name of the file you're uploading
:param f: A file object that can be read to upload to the server
:param metadata: *Optional* A bag of key/value pairs to be associated with this file (must be JSON encodable)

.. code:: python

    # Example of how to upload a file, first by saving it out from your library
    filename = '/tmp/model.npz'
    numpy.savez(filename, your_model_tensor)

    # Then sending it up to gradientzoo
    with open(filename, 'r') as f:
        zoo.upload_file(filename, f, {'loss': your_model_loss})
