try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Neural Machine Translator',
    'author': 'Armaan Puri',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'armaanpuri17@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'mactrans'
}

setup(**config)
