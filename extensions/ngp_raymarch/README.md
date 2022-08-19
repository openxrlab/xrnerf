# ngp_raymarch


## Install
安装cuda-extension，支持instant-ngp
```
cd extensions/ngp_raymarch
conda activate hashnerf
rm -rf build && clear && python setup.py build_ext --inplace \
2>&1 | tee build.log
python setup.py install
```
