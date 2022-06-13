from setuptools import Extension, dist, find_packages, setup

setup(name='openxrlab_xrnerf',
      description='Generic Framework for Nerf Algorithm',
      keywords='computer vision',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Utilities',
      ],
      zip_safe=False)
