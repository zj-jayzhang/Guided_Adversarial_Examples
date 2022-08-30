## Adversarial Examples Guided Imbalanced Learning 

This is the official PyTorch implementation of [Adversarial Examples for Good: Adversarial Examples Guided Imbalanced Learning](https://arxiv.org/abs/2201.12356) (ICIP 2022)

### How to use  

We provide several training examples with this repo:

- To train the ERM baseline (CE loss) on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```
![image](https://user-images.githubusercontent.com/33173674/187352063-7fd75a2c-d006-47e0-9c78-1bc01b74c1c7.png)



- To train the LDAM-DRW loss training on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
```
![image](https://user-images.githubusercontent.com/33173674/187352128-55243fde-55d5-4fb2-914e-d95db6a8a404.png)


- **Use our method to adjust the biased decision boundary**

```bash
python train_adv.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None \ 
        --resume checkpoint/cifar10_resnet32_CE_None_exp_0.01_0/ckpt.best.pth.tar
```
![image](https://user-images.githubusercontent.com/33173674/187352662-ee6cbd9b-9f99-4133-93e1-9e04fa6c513e.png)

![image](https://user-images.githubusercontent.com/33173674/187353220-70368fef-7707-440c-9cce-78f361e2e6eb.png)


> Note that we just simply finetune the biased model for several epochs, which is very efficient and effective.

### Citation

```
@article{zhang2022adversarial,
  title={Adversarial Examples for Good: Adversarial Examples Guided Imbalanced Learning},
  author={Zhang, Jie and Zhang, Lei and Li, Gang and Wu, Chao},
  journal={arXiv preprint arXiv:2201.12356},
  year={2022}
}
```

