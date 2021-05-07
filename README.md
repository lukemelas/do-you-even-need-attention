<div align="center">   

## Do You Even Need Attention? <br /> A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet
### [Paper/Report](https://arxiv.org/abs/2105.02723)
</div>
 
### TL;DR 
<img width="300" align="right" src="https://github.com/lukemelas/do-you-even-need-attention/releases/download/v0.0.1/do-you-even-need-attention.png" />
We replace the attention layer in a vision transformer with a feed-forward layer and find that it still works quite well on ImageNet. 

### Abstract
The strong performance of vision transformers on image classification and other vision tasks is often attributed to the design of their multi-head attention layers. However, the extent to which attention is responsible for this strong performance remains unclear. In this short report, we ask: is the attention layer even necessary? Specifically, we replace the attention layer in a vision transformer with a feed-forward layer applied over the patch dimension. The resulting architecture is simply a series of feed-forward layers applied over the patch and feature dimensions in an alternating fashion. In experiments on ImageNet, this architecture performs surprisingly well: a ViT/DeiT-base-sized model obtains 74.9\% top-1 accuracy, compared to 77.9\% and 79.9\% for ViT and DeiT respectively. These results indicate that aspects of vision transformers other than attention, such as the patch embedding, may be more responsible for their strong performance than previously thought. We hope these results prompt the community to spend more time trying to understand why our current models are as effective as they are.

### Note
This is concurrent research with [MLP-Mixer](https://arxiv.org/abs/2105.01601) from Google Research. The ideas are exacty the same, with the one difference being that they use (a lot) more compute. 

#### Pretrained models and logs

Here is a Weights and Biases report with the expected training trajectory: [W&B](https://wandb.ai/lukemelas2/deit-experiments/reports/Do-You-Even-Need-Attention---Vmlldzo2NjUxMzI?accessToken=8kebvweue0gd1s6qiav2orco97v85glogsi8i83576j42bb1g39e59px56lkk4zu)

| name | acc@1  | #params | url |
| --- | --- | --- | --- |
| FF-tiny | 61.4 | 7.7M | [model](https://github.com/lukemelas/do-you-even-need-attention/releases/download/v0.0.1/linear-tiny-checkpoint.pth) |
| FF-base | 74.9  | 62M | [model](https://github.com/lukemelas/do-you-even-need-attention/releases/download/v0.0.1/linear-base-checkpoint.pth) |
| FF-large | 71.4  | 206M | - |

Note: I haven't uploaded the `FF-Large` model because (1) it's over GitHub's file storage limit, and (2) I don't see why anyone would want it, given that it performs worse than the base model. That being said, if you want it, reach out to me and I'll send it to you. 

### How to train   

The model definition in `vision_transformer_linear.py` is designed to be run with the repo from DeiT, which is itself based on the wonderful `timm` package.

Steps:
 * Clone the DeiT repo and move the file into it
 ```bash
git clone https://github.com/facebookresearch/deit
mv vision_transformer_linear.py deit
cd deit
 ```
 
 * Add a line to import `vision_transformer_linear` in `main.py`. For example, add the following after the import statements (around line 27):
```diff
+ import vision_transformer_linear
```
 
 * Train: 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--master_port 10490 \
--use_env main.py \
--model linear_tiny \
--batch-size 128 \
--drop 0.1 \
--output_dir outputs/linear-tiny \
--data-path your/path/to/imagenet
```

#### Citation
If you build upon this idea, feel free to drop a citation (and also cite [MLP-Mixer](https://arxiv.org/abs/2105.01601)). 
```
@article{melaskyriazi2021doyoueven,
  title={Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet},
  author={Luke Melas-Kyriazi},
  journal=arxiv,
  year=2021
}
```
