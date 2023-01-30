# FAQ

## Outline

We list some common issues faced by many users and their corresponding solutions here.

- [FAQ](#faq)
  - [Outline](#outline)
  - [Installation](#installation)
  - [Data](#data)
  - [Training](#training)
  - [Testing](#testing)
  - [Deploying](#deploying)

Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

## Installation

- **"No module named 'mmcv'"**

    1. Install mmcv-full following the [installation instruction](https://mmcv.readthedocs.io/en/latest/#installation)


- **"No module named 'raymarch'"**

    1. Change workdir to extensions' directory using `cd extensions/ngp_raymarch`
    2. Compile cuda extensions using `rm -rf build && clear && python setup.py build_ext --inplace`
    3. Install cuda extensions using `python setup.py install`
