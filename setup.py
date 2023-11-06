from setuptools import setup, find_packages

# Python 3.11
setup(
    name='domain_invariant_generation_toy',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'pandas',
        'pytorch-lightning',
        'seaborn',
        'torch',
        'torchvision'
    ]
)