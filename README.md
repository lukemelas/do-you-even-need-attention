<div align="center">    
 
## Do You Even Need Attention? 
### A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet

[![Paper](http://img.shields.io/badge/Paper-B31B1B.svg)]()
</div>
 
### TL;DR 
We replace the attention layer in a vision transformer with a feed-forward layer and find that it still works quite well on ImageNet. 

### Abstract
Recent research in architecture design for computer vision has shown that transformers applied to sequences of unrolled image patches make for strong image classifiers. Much of this research focuses on modifying the transformer's attention layer, either to make it more efficient or better suited to the spacial structure of images. In this short report, we ask the question: is the attention layer even necessary? Specifically, we replace the attention layer with a second feed-forward layer over the patch features, resulting in an architecture  is simply a series of feed-forward networks applied over the patch and feature dimensions. In experiments on ImageNet, we show that a ViT-base-sized model obtains 74.896\% top-1 accuracy, rivaling a ResNet-50 (albeit with more parameters). Apart from its simplicity, this architecture does not appear to have any immediate practical advantages over a vision transformer with attention---it performs slightly worse and still requires $O(n^2)$ memory---but we hope the computer vision community will find it interesting nonetheless.  

### Note
This is concurrent research with [MLP-Mixer](https://arxiv.org/abs/2105.01601) from Google Research. The ideas are exacty the same, with the one difference being that they use (a lot) more compute. 

### How to train   

The model definition in `vision_transformer_linear.py` is designed to be run with the repo from DeiT, which is itself based on the wonderful `timm` package.

Steps:
 * Clone the DeiT repo 
 ```bash

 ```

#### Pretrained models and logs

Here is a Weights and Biases report with the expected training trajectory: [W&B](https://wandb.ai/lukemelas2/deit-experiments/reports/Do-You-Even-Need-Attention---Vmlldzo2NjUxMzI?accessToken=8kebvweue0gd1s6qiav2orco97v85glogsi8i83576j42bb1g39e59px56lkk4zu)

| name | acc@1  | #params | url |
| --- | --- | --- | --- |
| FF-tiny | 61.4 | 7.7M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| FF-base | 74.9  | 62M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| FF-large | 71.4  | 206M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) |


#### Citation   
```
@article{melaskyriazi2021doyoueven,
  title={Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet},
  author={Luke Melas-Kyriazi},
  journal=arxiv,
  year=2021
}
```   
