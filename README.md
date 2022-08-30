## Adversarial Examples Guided Imbalanced Learning 

### Training 

We provide several training examples with this repo:

- To train the ERM baseline (CE loss) on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```

- To train the LDAM-DRW loss training on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
```

- Use our method to adjust the biased decision boundary

```bash
python train_adv.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --resume checkpoint/cifar10_resnet32_CE_None_exp_0.01_0/ckpt.best.pth.tar
```
