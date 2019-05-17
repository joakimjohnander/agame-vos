# agame-vos
PyTorch implementation of the paper "A Generative Appearance Model for End-to-End Video Object Segmentation", including complete training code and trained models.

## Dependencies:
```bash
python (>= 3.5 or 3.6)
numpy
pytorch (>= 0.5 probably)
torchvision
pillow
tqdm
```

## Datasets utilized:
DAVIS
YouTubeVOS

## How to setup:
i) Install dependencies
ii) Clone this repo:
```bash
git clone https://github.com/joakimjohnander/agame-vos.git
```
iii) Download datasets
iv) Set up local_config.py to point to appropriate directories for saving and reading data

## How to run method on DAVIS and YouTubeVOS:
i) Download weights from TBD
ii) Run
```bash
python3 -u runfiles/main_runfile001.py --test
```

## How to train (and test) a model:
i) Run
```bash
python3 -u runfiles/main_runfile001.py --train --test
```

Most settings used for training and evaluation are set in your runfiles. Each runfile should correspond to a single experiment. I supplied an example runfile.

