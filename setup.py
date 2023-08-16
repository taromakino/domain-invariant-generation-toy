from setuptools import setup, find_packages


setup(
    name='disentangled_prediction',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'pandas',
        'pytorch-lightning',
        'torch',
        'torchvision'
    ]
)