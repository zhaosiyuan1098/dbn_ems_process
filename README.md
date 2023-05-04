# dbn_ems_process

基于深度信念网络/DBN，通过人体表面肌电信号完成动作分类：

* 使用快速傅里叶变换(**FFT**)完成时频变换
* 使用奇异谱分解(**SSA**)降噪
* 分别使用**XceptionTime**和**OmniScaleCNN**完成肌电信号的时域和频域分类
* 使用深度信念网络(**DBN**)融合XceptionTime和OmniScaleCNN结果，获取最终分类结果

## 环境

详见requirements.txt/environment.yaml


os              : Windows-10-10.0.22000-SP0
python          : 3.9.16
tsai            : 0.3.5
fastai          : 2.7.11
fastcore        : 1.5.29
torch           : 1.13.0
device          : 1 gpu (['NVIDIA GeForce GTX 1650'])
cpu cores       : 6
threads per cpu : 2
RAM             : 15.85 GB
GPU memory      : [4.0] GB

## 参考

[https://github.com/timeseriesAI/tsai](https://github.com/timeseriesAI/tsai)

[https://timeseriesai.github.io/tsai/](https://timeseriesai.github.io/tsai/)

[https://github.com/lutzroeder/Netron](https://github.com/lutzroeder/Netron)

[https://github.com/hfawaz/InceptionTime](https://github.com/hfawaz/InceptionTime)

Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J. & Petitjean, F. (2019). InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.

## TO DO

网络参数优化

代码优化

效果展示

## 效果展示

正确率：0.988
