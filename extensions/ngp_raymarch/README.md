# ngp_raymarch


## Install
build and install cuda-extension，to support instant-ngp
```
cd extensions/ngp_raymarch
rm -rf build && clear && python setup.py build_ext --inplace \
2>&1 | tee build.log
python setup.py install
```
