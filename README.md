# TransMatch
This repository contains the implementations of the paper 'TransMatch: A Transformer-based Multilevel Dual-Stream Feature Matching Network for Unsupervised Deformable Image Registration,' published in the IEEE Transactions on Medical Imaging (TMI) journal. 

[[Paper](https://ieeexplore.ieee.org/abstract/document/10158729/)] [[Code](https://github.com/tzayuan/TransMatch_TMI)]

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.1-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

## What's News
[4/2/2025] - **Cross-modal image registration supported now.**

[12/17/2023] - **The code has been re-reviewed and debugged, and it is now ready for direct training and inference on the provided LPBA40 example dataset.**
        
## Descriptions
This repository currently provides examples of implementing, training, and inferring with the core model code. For the 'transmatch' folder, its contents are almost identical to those in Models, except for the location of STN, which is specifically prepared for multimodal registration. It also includes guidance on running the code on sample datasets. Additionally, there are suggestions and code for visual analysis.

## Train and Infer Command
Before running the commands, please ensure that the dataset has been correctly placed. Taking the example of running the sample code on the LPBA40 dataset, ensure that the LPBA40 dataset is placed under ```../../../Dataset/LPBA40_delineation/```. This will ensure that the code can run directly without encountering any path-related errors. (Here, ```./``` refers to the directory path where ```Train.py``` and ```Infer.py``` are located.)

For Linux:

Train (mono-modality):  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Train.py```

Infer (mono-modality):  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Infer.py```

Train (multi-modality):  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Train_mm_srreg.py```

Infer (multi-modality):  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Infer_mm_srreg.py```

## Dataset
LPBA40 Datasets is uploded now.  [[LPBA40 datasets download link](https://drive.google.com/file/d/1mRmJpk06guietL3tUxpJjPYzEoJ0GLtm/view?usp=sharing)]

SR-REG Datasets and multimodal ckpt of SR-REG is uploede now. [[SR-REG datasets download link](https://drive.google.com/file/d/1YP0IOy3vthsC-QbDB4mbDPzJmyQ-ZsB4/view?usp=sharing)]  [[ckpt for SR-REG inference download link](https://drive.google.com/file/d/1Yrb3Ja5AkMQ0Ic14C6Tuzx2lZ5RvosSp/view?usp=sharing)]

Additionally, you can effortlessly train the code by customizing your datasets. Please be aware that you'll need to adjust the code file located at ```/utils/datagenerators_atlas.py``` to ensure the dataset is loaded in accordance with your specific dataset organization format.

## TODO
- [x] Core implementation code
- [x] Description of run script
- [x] Visualization code
- [x] Datasets url link (LPBA40 is uploaded)
- [x] Cross-modal image registration supported
- [ ] ~~Docker images & Google Colab Documents~~ (Pull requests are always welcome!)



## <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30"> Contact
Feel free to contact me if you have any further questions: snowbplus [AT] gmail [DOT] com.

## Citation
If you find this code is useful in your research, please consider to cite:
```
@article{chen2023transmatch,
  title={Transmatch: A transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration},
  author={Chen, Zeyuan and Zheng, Yuanjie and Gee, James C},
  journal={IEEE transactions on medical imaging},
  volume={43},
  number={1},
  pages={15--27},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgements
Thanks to Junyu Chen and Tony C. W. Mok for their guidance and help with the open source code.

