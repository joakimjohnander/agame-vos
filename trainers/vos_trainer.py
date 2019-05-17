import subprocess
from itertools import zip_longest
import glob

import torch
import torch.nn.functional as F
import torchvision as tv

import trainers
import utils

class VOSTrainer(object):
    def __init__(self, model, optimizer, objective_vos, lr_sched,
                 train_loader_vos, val_loader_vos,
                 use_gpu=True, workspace_dir=None, save_name=None,
                 checkpoint_interval=1, print_interval=25, debug=False):
        self._model = model
        self._optimizer = optimizer
        self._objective_vos = objective_vos
        self._lr_sched = lr_sched
        
        self._train_loader_vos = train_loader_vos
        self._val_loader_vos = val_loader_vos

        # Initialize statistics variables
        self._stats = {}
        if self._train_loader_vos is not None:
            self._stats['train_vos loss'] = utils.AverageMeter()
            self._stats['train_vos mIoU'] = utils.AverageMeter()
        if self._val_loader_vos is not None:
            self._stats['val_vos loss'] = utils.AverageMeter()
            self._stats['val_vos mIoU'] = utils.AverageMeter()

        self._use_gpu = use_gpu
        self._workspace_dir = workspace_dir
        self._save_name = save_name
        self._debug=debug
        self._checkpoint_interval = checkpoint_interval
        self._print_interval = print_interval

        self._epoch = 1
        if use_gpu:
            self._device = 'cuda'
        else:
            self._device = 'cpu'
        self._model.to(self._device)

    def train(self, max_epochs):
        for epoch in range(self._epoch, max_epochs+1):
            self._epoch = epoch
            self.train_epoch()
            if (self._epoch % self._checkpoint_interval == 0):
                print("Saving Checkpoint, current statistics are:")
                for key,val in self._stats.items():
                    strout = ["{:.3f}".format(elem) for elem in val.history]
                    print(key, strout, flush=True)
                self.save_checkpoint()
            elif all([self._stats['val_vos mIoU'].history[-1] > i
                      for i in self._stats['val_vos mIoU'].history[:-1]]):
                print("New best checkpoint, after epoch {}".format(self._epoch))
                self.save_checkpoint(alternative_name='best')
            self._lr_sched.step()
        print('Finished training!', flush=True)

    def train_epoch(self):
        """Do one epoch of training and validation."""
        self._model.train(True)
        self.cycle_dataset(mode='train')

        # Validation
        if self._val_loader_vos is not None:
            self._model.train(False)
            with torch.no_grad():
                self.cycle_dataset(mode='val')

        # Update all stat values
        for stat_value in self._stats.values():
            if isinstance(stat_value, utils.AverageMeter):
                stat_value.new_epoch()

    def cycle_dataset(self, mode):
        """Do a cycle of training or validation.
        Assumptions:
            loader outputs ((images, firstframe_segmentation, None), (perframe_labels, single_label))
            images: (batchsize,samplelen,nchannels,height,width) Tensor corresp to video
            firstframe_label: (batchsize,nclasses,height,width) Tensor with labels for the first frame
            None: gives space for internal states from the network
            perframe_labels: (batchsize,samplelen,height,width) Tensor with labels for all frames
            single_label: (batchsize,height,width) Tensor with pixel values as class
            model output: (batchsize,nclasses,height,width) Tensor corresp to frame segmentation"""
        if mode == 'train':
            loader_vos = self._train_loader_vos
        elif mode == 'val':
            loader_vos = self._val_loader_vos
        
        if loader_vos is None:
            loader_vos = []

        vos_miou_extremes = [(1.0,-1), (.0,-1)]
        for i, vos_data in enumerate(loader_vos):
            # Read data
            vos_images = vos_data['images'].to(self._device)
            vos_initseganno = vos_data['given_seganno'].to(self._device)
            vos_segannos = vos_data['segannos'].to(self._device)
                
            if mode == 'train':
                self._optimizer.zero_grad()

            # Model inference
            vos_out, _ = self._model(vos_images, vos_initseganno, None)
            if len(vos_out['logsegs']) != 1:
                B,_,H,W = vos_initseganno.size()
                raise NotImplementedError("Model seems to track multiple targets during training, ids {}".format(vos_out['logsegs'].keys()))

            # Calculate total loss
            loss = {}
            B,N,K,H,W = vos_out['logsegs'][1].size()
            loss['vos'] = self._objective_vos(vos_out['logsegs'][1].view(B*N,K,H,W),
                                              vos_segannos.view(B*N,H,W))
            if vos_out.get('coarse_logsegs') is not None:
                loss['vos_aux1'] = self._objective_vos(vos_out['coarse_logsegs'][1].view(B*N,K,H,W),
                                                       vos_segannos.view(B*N,H,W))
            total_loss = sum(loss.values())

            # Backpropagate
            if mode == 'train':
                total_loss.backward()
                self._optimizer.step()

            # Store vos loss and vos miou
            B,N,K,H,W = vos_out['logsegs'][1].size()
            loss['vos'] = loss['vos'].detach().to("cpu")
            self._stats[mode + '_vos loss'].update(loss['vos'].item(), B)
            vos_iou = utils.get_intersection_over_union(
                vos_out.get('logsegs')[1].view(B*N,K,H,W).detach(),
                vos_segannos.view(B*N,H,W).detach()).view(B, N, K) # Care only about channel 1 (target)
            tmp = vos_iou[:,:,1].mean(dim=1)
            if tmp.min() < vos_miou_extremes[0][0]: vos_miou_extremes[0] = (tmp.min(), i*B + tmp.argmin())
            if tmp.max() > vos_miou_extremes[1][0]: vos_miou_extremes[1] = (tmp.max(), i*B + tmp.argmax())

            vos_miou = vos_iou[:,:,1:].mean(dim=1).mean().to("cpu")
            self._stats[mode + '_vos mIoU'].update(vos_miou.item(), B)
            if loss.get('vos_aux1') is not None:
                loss['vos_aux1'] = loss['vos_aux1'].detach().to("cpu")


#            print(subprocess.check_output('nvidia-smi').decode('UTF-8'))
#            print(torch.cuda.memory_allocated() // 1000000)

            # Save some statistics
            if (i + 1) % self._print_interval == 0:
                if self._stats.get(mode+'_vos loss') is not None:
                    print("[{}: {}, {:5d}] Loss: {:.5f}".format(
                        mode, self._epoch,i+1, self._stats[mode+'_vos loss'].avg))

        # end for
        print("[{}: {}] Loss vos: {:.5f}".format(mode, self._epoch, self._stats[mode+'_vos loss'].avg))
        print("[{}: {}] mIoU vos: {:.5f}".format(mode, self._epoch, self._stats[mode+'_vos mIoU'].avg))
        print("Worst mIoU this batch was {:.3f} (idx {}), and best {:.3f} (idx {})".format(vos_miou_extremes[0][0],
                                                                                           vos_miou_extremes[0][1],
                                                                                           vos_miou_extremes[1][0],
                                                                                           vos_miou_extremes[1][1]))

    def save_checkpoint(self, alternative_name=None):
        """Saves a checkpoint of the network and other variables."""

        net_type = type(self._model).__name__
        state = {
            'epoch': self._epoch,
            'net_type': net_type,
            'net': self._model.state_dict(),
            'optimizer' : self._optimizer.state_dict(),
#            'lr_sched' : self._lr_sched.state_dict(),
            'stats' : self._stats,
            'use_gpu' : self._use_gpu,
        }

        if alternative_name is not None:
            file_path = '{}/{}_{}.pth.tar'.format(self._workspace_dir, self._save_name, alternative_name)
        elif self._save_name is None:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, net_type, self._epoch)
        else:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, self._save_name, self._epoch)
        torch.save(state, file_path)


    def load_checkpoint(self, checkpoint = None, verbose=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net_type = type(self._model).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            if self._save_name is None:
                checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._workspace_dir, net_type)))
            else:
                checkpoint_list = sorted(glob.glob('{}/{}_ep*.pth.tar'.format(self._workspace_dir, self._save_name)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        checkpoint_dict = torch.load(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        self._epoch = checkpoint_dict['epoch'] + 1
        self._model.load_state_dict(checkpoint_dict['net'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])
#        self._lr_sched.load_state_dict(checkpoint_dict['lr_sched'])
        self._stats = checkpoint_dict['stats']
        self._use_gpu = checkpoint_dict['use_gpu']

        if verbose:
            print("Loaded: {}".format(checkpoint_path))
        

