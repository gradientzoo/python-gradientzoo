from __future__ import print_function

import os

from .client import Gradientzoo, StatusCodeError, NotFoundError

__all__ = ['Gradientzoo', 'StatusCodeError', 'NotFoundError',
           'KerasGradientzoo', 'GradientzooCallback']

try:
    from keras import __version__ as keras_version
    from keras.callbacks import Callback
    has_keras = True
except ImportError:
    has_keras = False

if has_keras:
    class GradientzooCallback(Callback):

        def __init__(self, client, model, after_batches=None, after_epochs=1):
            super(GradientzooCallback, self).__init__()

            self.client = client
            self.model = model
            self.after_batches = after_batches
            self.after_epochs = after_epochs

            self.batch = 0
            self.epoch = 0

        def _meta(self, logs):
            dct = {}
            if logs:
                if 'acc' in logs:
                    dct['acc'] = float(logs['acc'])
                if 'loss' in logs:
                    dct['loss'] = float(logs['loss'])
            return dct

        def on_batch_end(self, batch, logs=None):
            if not self.after_batches:
                return
            self.batch += 1
            if self.batch == self.after_batches:
                self.batch = 0
                try:
                    self.client.save(self.model, self._meta(logs))
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception, e:
                    print('Could not save weights: {}'.format(e))

        def on_epoch_end(self, epoch, logs=None):
            if not self.after_epochs:
                return
            self.epoch += 1
            if self.epoch == self.after_epochs:
                self.epoch = 0
                try:
                    self.client.save(self.model, self._meta(logs))
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception, e:
                    print('Could not save weights: {}'.format(e))

    class KerasGradientzoo(Gradientzoo):
        framework = 'keras'
        framework_version = keras_version

        def load(self, model, filename='model.h5', id=None, dir=None):
            # Download the file into the directory
            filepath, file_model = self.download_file(filename, id, dir=dir)

            # Actually load the model weights
            model.load_weights(filepath)

            return file_model

        def save(self, model, filename='model.h5', metadata=None, dir=None):
            # Figure out the path to save the file to temporarily
            if not dir:
                dir = self.default_dir
            filepath = os.path.join(dir, filename)

            # Save the file
            model.save_weights(filepath, overwrite=True)

            # Upload the file
            with open(filepath, 'r') as f:
                data = self.upload_file(filename, f, metadata).json()

            # Move the file to the correct spot in the cache atomically
            os.rename(filepath,
                      os.path.join(dir, data['file']['id'] + '_' + filename))

        def make_callback(self, model, after_batches=None, after_epochs=1):
            return GradientzooCallback(self, model, after_batches, after_epochs)
else:
    class GradientzooCallback(object):
        pass

    class KerasGradientzoo(object):
        pass
