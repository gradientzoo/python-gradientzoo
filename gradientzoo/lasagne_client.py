from __future__ import print_function

import os

import numpy as np

from .client import Gradientzoo, StatusCodeError, NotFoundError

__all__ = ['Gradientzoo', 'StatusCodeError', 'NotFoundError',
           'LasagneGradientzoo']

try:
    import lasagne
    from lasagne import __version__ as lasagne_version
    has_lasagne = True
except ImportError:
    has_lasagne = False

if has_lasagne:
    class LasagneGradientzoo(Gradientzoo):
        framework = 'lasagne'
        framework_version = lasagne_version

        def load(self, network, filename='model.npz', id=None, dir=None):
            # Download the file into the directory
            filepath, f = self.download_file(filename, id, dir=dir)

            # Load the model weights and process them
            with np.load(filepath) as data_file:
                param_values = [
                    data_file['arr_%d' % i] for i in range(len(data_file.files))]

            # Pass them into lasagne
            lasagne.layers.set_all_param_values(network, param_values)

            return f

        def save(self, network, metadata=None, filename='model.npz', dir=None):
            # Figure out the path to save the file to temporarily
            if not dir:
                dir = self.default_dir
            filepath = os.path.join(dir, filename)

            # Save the file
            np.savez(filepath, *lasagne.layers.get_all_param_values(network))

            # Upload the file
            with open(filepath, 'r') as f:
                self.upload_file(filename, f, metadata)
else:
    class LasagneGradientzoo(object):
        pass
