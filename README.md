# Antipodes of Label Differential Privacy: PATE and ALIBI

This repository is the official implementation of [Antipodes of Label Differential Privacy: PATE and ALIBI](https://arxiv.org/abs/2106.03408).

## Citation
```bibtex
@misc{malek2021antipodes,
      title={Antipodes of Label Differential Privacy: {PATE} and {ALIBI}},
      author={Mani Malek and Ilya Mironov and Karthik Prasad and Igor Shilov and Florian Tram{\`e}r},
      year={2021},
      eprint={2106.03408},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
The above command assumes a Linux environment with CUDA support. Please refer to https://pytorch.org/ for your specific environment.

If you're training PATE with CIFAR100, you'll also need to install `apex` manually:
```setup
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```

## Training

### PATE
PATE model is trained in 3 stages. Below are example commands with hyperparameters we used.

#### Stage 1: Train teacher ensemble
Optionally add `-N 1000` for canary runs
##### CIFAR10
```commandline
python train_teacher.py --dataset cifar10 --n_teachers 800 --teacher-id 0 --epochs 40
```

##### CIFAR100
```commandline
python train_teacher.py --dataset cifar100 --n_teachers 100 --teacher-id 0 --epochs 125 --weight_decay 0.001 --width 8 --amp true --opt_level O2
```

#### Stage 2: Aggregate votes
Once all teachers are trained, we need to aggregate all votes into a single file

```commandline
python aggregate_votes.py --n_teachers 800
```

#### Stage 3: Train student

##### CIFAR10

```commandline
python train_student.py --dataset cifar10 --n_samples 250 --n_teachers 800 --selection_noise 800 --result_noise 500 --noise.threshold 400 --epochs 200
```

##### CIFAR100
```commandline
python train_student.py --dataset cifar100 --n_samples 1000 --n_teachers 100 --epochs 125 --weight_decay 0.001 --width 8 --amp true --opt_level O2
```

### ALIBI
`train_cifar_alibi.py` implements the training and evaluation of ResNet on CIFAR.
Run the following to explore the arguments (dataset, model, architecture, noising mechanism, post-processing mode, hyperparameters, training knobs, etc.) that can be set during the runs.
```commandline
python train_cifar_alibi.py --help
```
Most notably,
* Use `--dataset "CIFAR10"` and `--dataset "CIFAR100"` to train on CIFAR-10 and CIFAR-100 respectively.
* Use `--arch "resnet"` and `--arch "wide-resnet"` to train on ResNet-18 and Wide-Resnet18-100 respectively.
* Use `--canary 1000` to train with 1000 mislabeled "canaries".
* In our paper, we used `--seed 11337`

> NOTE: Tables 3 and 4 summarize hyperparameters for PATE-FM and ALIBI respectively.

## Memorization attacks on trained models
To reproduce the memorization attack on our trained models, run
```commandline
cd memorization_attack
python attack.py
```

## Results
Results are summarized in Tables 1 and 2 of our paper.


## License
The majority of facebookresearch/label_dp_antipodes is licensed under CC-BY-NC, however portions of the project are available under separate license terms: kekmodel/FixMatch-pytorch, kuangliu/pytorch-cifar, and facebookresearch/label_dp_antipodes/memorization_attack are licensed under MIT license, tensorflow/privacy is licensed under Apache-2.0 license.


## Acknowledgements
* Our FixMatch implementation is heavily based on [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch)
* PATE privacy accounting is adopted from [TensorFlow Privacy](https://github.com/tensorflow/privacy/tree/master/research/pate_2018/ICLR2018)
* ResNet implementation for CIFAR is borrowed from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
