from setuptools import setup, find_packages


setup(
    name='domain_invariant_generation_toy',
    packages=find_packages(),
    python_requires='3.11',
    install_requires=[
        'matplotlib',
        'pandas',
        'pytorch-lightning',
        'torch',
        'torchvision'
    ]
)