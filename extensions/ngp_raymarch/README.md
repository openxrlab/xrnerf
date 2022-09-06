# ngp_raymarch


## Install
build and install cuda-extension，to support instant-ngp
```
cd extensions/ngp_raymarch
rm -rf build && clear && python setup.py build_ext --inplace \
2>&1 | tee build.log
python setup.py install
```

## Notice
* This code mainly based on [instance-ngp](https://github.com/NVlabs/instant-ngp) code modification
* This code's license belongs to [instance-ngp](https://github.com/NVlabs/instant-ngp/blob/master/LICENSE.txt)
* If you found this code useful, please cite [instance-ngp](https://github.com/NVlabs/instant-ngp#license-and-citation)
* We appreciate [instance-ngp](https://github.com/NVlabs/instant-ngp) for their cool code implementation
