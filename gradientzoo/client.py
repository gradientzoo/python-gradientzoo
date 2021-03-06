from __future__ import print_function

from . import __version__ as gradientzoo_version

import json
import os

import requests

__all__ = ['StatusCodeError', 'NotFoundError', 'Gradientzoo']

DEFAULT_DIR = os.path.realpath(os.path.expanduser('~/.gradientzoo/cache'))
API_BASE = 'https://api.gradientzoo.com'


class StatusCodeError(ValueError):

    def __init__(self, status_code, *args, **kwargs):
        self.status_code = status_code
        super(StatusCodeError, self).__init__(*args, **kwargs)


class NotFoundError(StatusCodeError):

    def __init__(self, *args, **kwargs):
        super(NotFoundError, self).__init__(404, *args, **kwargs)


class Gradientzoo(object):
    framework = 'python'
    framework_version = gradientzoo_version

    def __init__(self, username=None, model_slug=None, api_base=API_BASE,
                 auth_token_id=os.environ.get('GRADIENTZOO_AUTH_TOKEN_ID'),
                 default_dir=DEFAULT_DIR):
        if not username:
            raise ValueError(
                'Must specify the Gradientzoo model e.g. "ericflo/mnist_cnn"')

        self.api_base = api_base
        self.session = requests.Session()
        self.session.headers.update(self.default_headers())
        self.auth_token_id = auth_token_id
        self.username, self.slug = self._parse_username_slug(username,
                                                             model_slug)
        self.default_dir = default_dir
        try:
            os.makedirs(default_dir)
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            pass

    def default_headers(self):
        return {
            'X-Gradientzoo-Client-Name': 'python-gradientzoo',
            'X-Gradientzoo-Framework-Version': self.framework_version,
        }

    @property
    def auth_token_id(self):
        return self.session.headers.get('X-Auth-Token-Id')

    @auth_token_id.setter
    def auth_token_id(self, value):
        if not value:
            try:
                del self.session.headers['X-Auth-Token-Id']
            except KeyError:
                pass
        self.session.headers.update({'X-Auth-Token-Id': value})

    def _parse_username_slug(self, username, slug):
        un, _, sl = username.partition('/')
        return (un, sl or slug)

    def full_url(self, path):
        slash_end = self.api_base[-1] == '/'
        slash_begin = path[0] == '/'
        if slash_end and slash_begin:
            return self.api_base + path[1:]
        if not slash_end and not slash_begin:
            return self.api_base + '/' + path
        return self.api_base + path

    def post(self, *args, **kwargs):
        args_list = [self.full_url(args[0])] + list(args[1:])
        return self.session.post(*args_list, **kwargs)

    def get(self, *args, **kwargs):
        args_list = [self.full_url(args[0])] + list(args[1:])
        return self.session.get(*args_list, **kwargs)

    def _file_path(self, filename=None, id=None):
        if id:
            return 'file-id/{}'.format(id)
        return 'file/{}/{}/{}/{}'.format(
            self.username,
            self.slug,
            self.framework,
            filename,
        )

    def upload_file(self, filename, f, metadata=None):
        if not metadata:
            metadata = {}
        return self.post(self._file_path(filename),
                         data={'metadata': json.dumps(metadata)},
                         files={'file': f})

    def download_file(self, filename, id=None, dir=None, chunk_size=1024):
        # If they didn't specify the dir, use the default one and make sure it
        # actually exists
        if not dir:
            dir = self.default_dir

        # Get the URL from gradientzoo
        r = self.get(self._file_path(filename, id))
        if r.status_code != 200:
            err = '(no error message available)'
            try:
                err = r.json().get('error')
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass
            if r.status_code == 404:
                raise NotFoundError(
                    'Not found [{}]: {}'.format(filename, err))
            raise StatusCodeError(r.status_code,
                                  'Got bad status code {}: {}'.format(r, err))

        data = r.json()

        end_filepath = os.path.join(dir, data['file']['id'] + '_' + filename)
        tmp_filepath = end_filepath + '.tmp'

        # If we don't have a copy downloaded
        if not os.path.exists(end_filepath):
            # Download the content from the CDN url
            r = requests.get(data['url'], stream=True)
            if r.status_code != 200:
                if r.status_code == 404:
                    raise NotFoundError(
                        'Could not find CDN file {}'.format(filename))
                raise StatusCodeError(r.status_code,
                                      'Got bad CDN status code {}'.format(r))

            # Read it all to the file in the directory specified
            with open(tmp_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size):
                    f.write(chunk)
                f.flush()

            # Rename the file to the proper destination atomically
            os.rename(tmp_filepath, end_filepath)

        # Return the file path
        return end_filepath, data.get('file', {})
