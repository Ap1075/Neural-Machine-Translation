try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Neural Machine Translator',
    'long_description': 'Machine translation between source and target launguages using \
                        an LSTM based model with pre trained\
                        word embeddings for source language.',
    'author': 'Armaan Puri',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'armaanpuri17@gmail.com',
    'version': '0.1',
    'install_requires': ["keras", "numpy", "tensorflow_gpu", "matplotlib"],
    'packages': ["mactrans"],
    'scripts': ["bin/train_mt.py", "bin/predict_mt.py"],
    'name': 'mactrans'
}

setup(**config)
