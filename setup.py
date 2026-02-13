from setuptools import setup, find_packages

setup(
    name='fastforest',
    version='1.0',
    packages=find_packages(),
    description='MABSPLIT Boosting extension of FastForest',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ThrunGroup/FastForest',
    project_urls={
        'Base Project': 'https://github.com/ThrunGroup/FastForest',
        'Attribution': 'ATTRIBUTION.md',
    },
    note='Built upon FastForest (https://github.com/ThrunGroup/FastForest)'
)
