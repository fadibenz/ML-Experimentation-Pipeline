from setuptools import setup, find_packages

setup(
    name='ML-Experimentation-Pipeline',
    version='0.1.0',
    description='',
    author='Fadi Benzaima',
    packages=find_packages(where='ML-Experementation-Pipeline'),
    package_dir={'': 'ML-Experementation-Pipeline'},
    install_requires=[
        'torch',
        'torchvision',
        'optuna',
        'wandb',
    ],
)