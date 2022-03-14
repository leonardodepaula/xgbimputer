from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='xgbimputer',
    version='0.2.0',
    author='Leonardo de Paula Liebscher',
    author_email='<leonardopx@gmail.com>',
    description='Extreme Gradient Boosting imputer for Machine Learning.',
    long_description_content_type='text/markdown',
    long_description=long_description,
    url='https://github.com/leonardodepaula/xgbimputer',
    packages=find_packages(),
    install_requires=['numpy>=1.21', 'scikit-learn>=1.0', 'xgboost>=1.5'],
    keywords=['python', 'machine learning', 'missing values', 'imputation'],
    classifiers=[
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python',
      'Development Status :: 4 - Beta'
    ]
)