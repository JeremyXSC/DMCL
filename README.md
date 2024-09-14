# Deep Multimodal Collaborative Learning for Polyp Re-Identification

## Introduction

we propose a novel Deep Multimodal Collaborative Learning framework named DMCL for polyp re-identification, which can effectively encourage modality collaboration and reinforce generalization capability in medical scenarios. On the basis of it, a dynamic multimodal feature fusion strategy is introduced to leverage the optimized multimodal representations for multimodal fusion via end-to-end training. Experiments on the standard benchmarks show the benefits of the multimodal setting over state-of-the-art unimodal ReID models, especially when combined with the specialized multimodal fusion strategy.

<img src='figs/DMCL.png'/>

### News
- Support Market1501, DukeMTMC-reID, CUHK03 and Colo-Pair datasets.

## Installation

TorchMultimodal requires Python >= 3.7. The library can be installed with or without CUDA support.
The following assumes conda is installed.

## Instruction
Modify the data path of the code according to your dataset, then perform the following command:
```
bash trainer.sh
```
'train_fold' represents the n fold Cross Validation.
 
## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## Acknowledgments
This work was supported by the National Natural Science Foundation of China under Projects (Grant No. 62301315).
If you have further questions and suggestions, please feel free to contact us (xiangsuncheng17@sjtu.edu.cn).

If you find this code useful in your research, please consider citing:
```
@article{xiang2024deep,
  title={Deep Multimodal Collaborative Learning for Polyp Re-Identification},
  author={Xiang, Suncheng and Li, Jincheng and Zhang, Zhengjie and Cai, Shilun and Guan, Jiale and Qian, Dahong},
  journal={arXiv preprint arXiv:2408.05914},
  year={2024}
}
```
