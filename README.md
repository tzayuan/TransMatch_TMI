# TransMatch_TMI
This is a repository of the implementations of manuscript "TransMatch: A Transformer-based Multilevel Dual-Stream Feature Matching Network for Unsupervised Deformable Image Registration", which is submitted to the [IEEE TMI journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=42).

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.5-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
        
## Descriptions
The repository currently contains all the core implementation code. File descriptions, datasets, usage, and some visualization code are being uploaded.

## Train and Infer Command
For Linux:

Train:  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python train.py```

Infer:  
```export CUDA_VISIBLE_DEVICES=0``` (If needed)  
```python infer.py```

## Dataset
Datasets is uploding now from my Synology NAS, upload expected to be completed by the end of June. You can now easily train the code by organizing your own datasets. Note that the code file located in ```/Model/datagenerators_atlas.py``` needs to be modified so that the dataset is loaded to match your dataset organization format.

## TODO
- [x] Core implementation code
- [x] Description of run script
- [x] visualization code
- [ ] Datasets url link (upload expected to be completed by the end of June)
- [ ] Checkpoint url link (upload expected to be completed by the end of June)



## Concact
Please feel free to concact me if you have any further questions: snowbplus [AT] gmail [DOT] com


## Acknowledgements
Thanks to Junyu Chen and Tony C. W. Mok for their guidance and help with the open source code.

