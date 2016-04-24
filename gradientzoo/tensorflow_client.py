from __future__ import print_function

import os

from .client import Gradientzoo, StatusCodeError, NotFoundError

__all__ = ['Gradientzoo', 'StatusCodeError', 'NotFoundError',
           'TensorflowGradientzoo']

try:
    from tensorflow import __version__ as tf_version
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

if has_tensorflow:
    class TensorflowGradientzoo(Gradientzoo):
        framework = 'tensorflow'
        framework_version = tf_version

        def __init__(self, *args, **kwargs):
            self.saver_kwargs = kwargs.pop('saver_kwargs', {})
            self._saver = None
            super(TensorflowGradientzoo, self).__init__(*args, **kwargs)

        @property
        def saver(self):
            if not self._saver:
                self._saver = tf.train.Saver(**self.saver_kwargs)
            return self._saver

        def load(self, session, filename='model.ckpt', id=None, dir=None):
            # Download the file into the directory
            filepath, file_model = self.download_file(filename, id, dir=dir)

            # Actually load the model weights
            self.saver.restore(session, filepath)

            return file_model

        def save(self, session, metadata=None, filename='model.ckpt', dir=None):
            # Figure out the path to save the file to temporarily
            if not dir:
                dir = self.default_dir
            filepath = os.path.join(dir, filename)

            # Save the file
            self.saver.save(session, filepath)

            # Upload the file
            with open(filepath, 'r') as f:
                data = self.upload_file(filename, f, metadata).json()

            # Move the file to the correct spot in the cache atomically
            os.rename(filepath,
                      os.path.join(dir, data['file']['id'] + '_' + filename))
else:
    class TensorflowGradientzoo(object):
        pass
