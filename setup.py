import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='knnor_reg',
      version='0.0.2',
      description='Generic python library to perform augmentation of regression data.',
      long_description='''A generic package to help data scientists balance their regression dataset by increasing the datapoints for underrepresented data.
      pip install knnor-reg
      '''
                  ,
      url='',
      author='Ashhadul Islam, Sameer Brahim Belhaouari',
      author_email='ashhadulislam@gmail.com, samir.brahim@gmail.com',
      keywords='Regression, Data Augmentation, Imabalnced Data',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[            
            "numpy",
            "scikit-learn",
            "matplotlib",
            "pandas",
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False
)