import os
import sys
import json
import random
from collections import OrderedDict
from optparse import OptionParser

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from torch.optim import lr_scheduler
from PIL import Image

import dataset_loaders
from dataset_loaders import IMAGENET_MEAN, IMAGENET_STD, DAVIS17V2, YTVOSV2, LabelToLongTensor
import evaluation
import models
import trainers
import utils
from local_config import config

parser = OptionParser()
parser.add_option("--train", action="store_true", dest="train", default=None)
parser.add_option("--test", action="store_true", dest="test", default=None)
(options, args) = parser.parse_args()


##################
# Training
##################
def train_alpha(model):
    num_epochs = 80
    batch_size = 4
    nframes = 8
    nframes_val = 32

    size = (240, 432)
    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.BILINEAR),
             tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)
    def label_read(path):
        if os.path.exists(path):
            pic = Image.open(path)
            transform = tv.transforms.Compose(
                [tv.transforms.Resize(size, interpolation=Image.NEAREST),
                 LabelToLongTensor()])
            label = transform(pic)
        else:
            label = torch.LongTensor(1,*size).fill_(255) # Put label that will be ignored
        return label
    def random_object_sampler(lst):
        return [random.choice(lst)]
    def deterministic_object_sampler(lst):
        return [lst[0]]
    train_transform = dataset_loaders.JointCompose([dataset_loaders.JointRandomHorizontalFlip(),
                                                    dataset_loaders.JointRandomScale()])

    train_set = torch.utils.data.ConcatDataset([
        DAVIS17V2(config['davis17_path'], '2017', 'train', image_read, label_read, train_transform, nframes,
                  random_object_sampler, start_frame='random'),
    ])
    val_set = YTVOSV2(config['ytvos_path'], 'train', 'val_joakim', 'JPEGImages', image_read, label_read, None,
                      nframes_val, deterministic_object_sampler, start_frame='first')
    sampler = torch.utils.data.WeightedRandomSampler(len(train_set)*[1,], 2600, replacement=True)
    train_loader = DataLoader(train_set, sampler=sampler, batch_size=batch_size, num_workers=11)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=11)
    print("Sets initiated with {} (train) and {} (val) samples.".format(len(train_set), len(val_set)))

    objective = nn.NLLLoss(ignore_index=255).cuda()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                 lr=1e-4, weight_decay=1e-5)

    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, .95)

    trainer = trainers.VOSTrainer(
        model, optimizer, objective, lr_sched,
        train_loader, val_loader,
        use_gpu=True, workspace_dir=config['workspace_path'],
        save_name=os.path.splitext(os.path.basename(__file__))[0]+"_alpha",
        checkpoint_interval=80, print_interval=25, debug=False)
    trainer.load_checkpoint()
    trainer.train(num_epochs)

def train_beta(model):
    print("Starting initial training (with cropped images)")
    num_epochs = 100
    batch_size = 2
    nframes = 14
    nframes_val = 32

    size = (480, 864)
    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.BILINEAR),
             tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)
    def label_read(path):
        if os.path.exists(path):
            pic = Image.open(path)
            transform = tv.transforms.Compose(
                [tv.transforms.Resize(size, interpolation=Image.NEAREST),
                 LabelToLongTensor()])
            label = transform(pic)
        else:
            label = torch.LongTensor(1,*size).fill_(255) # Put label that will be ignored
        return label
    def random_object_sampler(lst):
        return [random.choice(lst)]
    def deterministic_object_sampler(lst):
        return [lst[0]]
    train_transform = dataset_loaders.JointCompose([dataset_loaders.JointRandomHorizontalFlip()])

    train_set = torch.utils.data.ConcatDataset([
        DAVIS17V2(config['davis17_path'], '2017', 'train', image_read, label_read, train_transform, nframes,
                  random_object_sampler, start_frame='random'),
    ])
    val_set = YTVOSV2(config['ytvos_path'], 'train', 'val_joakim', 'JPEGImages', image_read, label_read, None,
                      nframes_val, deterministic_object_sampler, start_frame='first')

    sampler = torch.utils.data.WeightedRandomSampler(len(train_set)*[1,], 118, replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=11)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=11)
    print("Sets initiated with {} (train) and {} (val) samples.".format(len(train_set), len(val_set)))

    objective = nn.NLLLoss(ignore_index=255).cuda()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                 lr=1e-5, weight_decay=1e-6)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, .985)

    trainer = trainers.VOSTrainer(
        model, optimizer, objective, lr_sched,
        train_loader, val_loader,
        use_gpu=True, workspace_dir=config['workspace_path'],
        save_name=os.path.splitext(os.path.basename(__file__))[0]+"_beta",
        checkpoint_interval=100, print_interval=25, debug=False)
    trainer.load_checkpoint()
    trainer.train(num_epochs)

    
##################
# Testing
##################

def test_model(model, debug_sequences_dict=None, save_predictions=False, prediction_path=None):
    nframes = 128
    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)
    def label_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [LabelToLongTensor()])
        label = transform(pic)
        return label
    datasets = {
        'DAVIS16_train': DAVIS17V2(config['davis17_path'], '2016', 'train', image_read, label_read, None, nframes),
        'DAVIS16_val': DAVIS17V2(config['davis17_path'], '2016', 'val', image_read, label_read, None, nframes),
        'DAVIS17_val': DAVIS17V2(config['davis17_path'], '2017', 'val', image_read, label_read, None, nframes),
        'YTVOS_jval': YTVOSV2(config['ytvos_path'], 'train', 'val_joakim', 'JPEGImages', image_read, label_read,
                              None, nframes),
        'YTVOS_val': YTVOSV2(config['ytvos_path'], 'valid', None, 'JPEGImages', image_read, label_read,
                             None, nframes)
    }
    multitarget_sets = ('DAVIS17_val', 'YTVOS_jval', 'YTVOS_val')
    
    if debug_sequences_dict is None:
        debug_sequences_dict = {key: () for key in datasets.keys()}

    for key, dataset in datasets.items():
        if key == 'YTVOS_val':
            evaluator = evaluation.VOSEvaluator(dataset, 'cuda', 'all', True, False, debug_sequences_dict.get(key))
        elif key == 'DAVIS17_val':
            evaluator = evaluation.VOSEvaluator(dataset, 'cuda', 'all', True, True, debug_sequences_dict.get(key))
        else:
            evaluator = evaluation.VOSEvaluator(dataset, 'cuda', 'all', False, True, debug_sequences_dict.get(key))
        result_fpath = os.path.join(config['output_path'], os.path.splitext(os.path.basename(__file__))[0])

        if key in multitarget_sets: # Only apply the multitarget aggregation if we have mult. targets
            model.update_with_softmax_aggregation = True
        else:
            model.update_with_softmax_aggregation = False
        evaluator.evaluate(model, os.path.join(result_fpath, key))

#debug_sequences_dict = {'DAVIS16 (train)':(), 'DAVIS16 (val)':('bmx-trees','drift-straight','motocross-jump','soapbox'), 'VOT16 (segm)':('ball1','bmx','book'), 'YTVOS':('97ab569ff3','c557b69fbf','8ea6687ab0','6cccc985e0')}
debug_sequences_dict = {'DAVIS16_train':(), 'DAVIS16_val':(), 'VOT16_segm':(), 'YTVOS_val':()}


def main():
    print("Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__))

    torch.cuda.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = models.AGAME(
        backbone=('resnet101s16', (True, ('layer4',),('layer4',),('layer2',),('layer1',))),
        appearance=('GaussiansAgame', (2048, 512, .1, -2)),
        dynamic=('RGMPLike', ([('ConvRelu', (2*2048+2*2,512,1,1,0)),
                               ('DilationpyramidRelu',(512,512,3,1,(1,3,6),(1,3,6))),
                               ('ConvRelu',(1536,512,3,1,1))],)),
        fusion=('FusionAgame',
                ([('ConvRelu',(516,512,3,1,1)), ('ConvRelu',(512,128,3,1,1))],
                 [('Conv', (128, 2, 1, 1, 0))])),
        segmod=('UpsampleAgame', ({'s8':512,'s4':256}, 128)),
        update_with_fine_scores=False, update_with_softmax_aggregation=False, process_first_frame=True,
        output_logsegs=True, output_coarse_logsegs=True, output_segs=False)
    print("Network model {} loaded, (size: {})".format(model.__class__.__name__, utils.get_model_size_str(model)))

    if options.train is not None:
        model.update_with_fine_scores = False
        model.update_with_softmax_aggregation = False
        model.process_first_frame = False
        model.output_logsegs = True
        model.output_coarse_logsegs = True
        model.output_segs = False
        train_alpha(model)
        model.output_coarse_logsegs = False
        train_beta(model)
    if options.test is not None:
        model.update_with_fine_scores = True
        model.update_with_softmax_aggregation = True
        model.process_first_frame = False
        model.output_logsegs = False
        model.output_coarse_logsegs = False
        model.output_segs = True
        runfile_name = os.path.splitext(os.path.basename(__file__))[0]
        checkpoint_basename = config['workspace_path'] + runfile_name
        model.load_state_dict(torch.load(checkpoint_basename + '_beta_best.pth.tar')['net'])
        test_model(model, debug_sequences_dict=debug_sequences_dict, save_predictions=False,
                   prediction_path=config['output_path'] + runfile_name + '_beta_best/')

if __name__ == '__main__':
    main()
