from setuptools import setup

setup(name='f110_gym',
      version='0.2',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      install_requires=[
            'gym==0.18.0', 'numpy', 'Pillow', 'scipy', 'numba', 'pyyaml']
      )