# TransMatch_TMI
This is a repository of the implementations of paper "TransMatch: A Transformer-based Multilevel Dual-Stream Feature Matching Network for Unsupervised Deformable Image Registration", which is published in [IEEE TMI journal](https://ieeexplore.ieee.org/abstract/document/10158729/).

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-2.1-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

## News
[12/17/2023] - **The code has been re-reviewed and debugged, and it is now ready for direct training and inference on the provided LPBA40 example dataset.**
        
## Descriptions
The repository currently contains all the core implementation code. File descriptions, datasets, usage, and some visualization code are being uploaded.

## Train and Infer Command
For Linux:

Train:  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Train.py```

Infer:  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python Infer.py```

## Dataset
LPBA40 Datasets is uploded now.  [[LPBA40 datasets download link](https://drive.google.com/file/d/1mRmJpk06guietL3tUxpJjPYzEoJ0GLtm/view?usp=sharing)]

Besides, you can now easily train the code by organizing your own datasets. Note that the code file located in ```/utils/datagenerators_atlas.py``` needs to be modified so that the dataset is loaded to match your dataset organization format.

## TODO
- [x] Core implementation code
- [x] Description of run script
- [x] Visualization code
- [x] Datasets url link (LPBA40 is uploaded)
- [ ] Docker images & Google Colab Documents 



## <img src="https://raw.githubusercontent.com/iampavangandhi/iampavangandhi/master/gifs/Hi.gif" width="30"> Contact
Please feel free to contact me if you have any further questions: snowbplus [AT] gmail [DOT] com


## Acknowledgements
Thanks to Junyu Chen and Tony C. W. Mok for their guidance and help with the open source code.

