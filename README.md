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
1. Install dependencies
2. Clone this repo:
```bash
git clone https://github.com/joakimjohnander/agame-vos.git
```
3. Download datasets
4. Set up local_config.py to point to appropriate directories for saving and reading data

## How to run method on DAVIS and YouTubeVOS:
1. Download weights from https://drive.google.com/file/d/1lVv7n0qOtJEPk3aJ2-KGrOfYrOHVnBbT/view?usp=sharing
2. Run
```bash
python3 -u runfiles/main_runfile001.py --test
```

## How to train (and test) a model:
1. Run
```bash
python3 -u runfiles/main_runfile001.py --train --test
```

Most settings used for training and evaluation are set in your runfiles. Each runfile should correspond to a single experiment. I supplied an example runfile.

