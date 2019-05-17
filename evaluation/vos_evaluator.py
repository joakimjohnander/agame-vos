import os
import time
import json

import torch

import utils


class VOSEvaluator(object):
    """Provides an evaluate function that evaluates a given model, on the dataset specified during initialization
    of the evaluator. On evaluate call, the model is run on sequences_to_eval. The predictions made by the model
    are compared to the labels provided by the dataset if the dataset provides labels, and if so the dataset mmIoU
    is calculated (and stored). If desired, the predictions made can be saved for visualization or metric calculation
    by external means.
    Args:
        dataset (VOS-compatible torch.utils.data.dataset)
        device (torch.device): 'cpu', 'cuda', 'cuda:0', ..., The device where model and data is put
        sequences_to_eval: 'all' or list of sequence names that are to be evaluated
    """
    def __init__(self, dataset, device='cuda', sequences_to_eval='all',
                 save_predicted_segmentations=set(), calculate_measures=set(), debug_sequences=(),
                 skip_existing=False):
        self._dataset = dataset
        self._device = device

        if debug_sequences is None:
            debug_sequences = ()

        # To cope with legacy runfiles
        if calculate_measures is False:
            calculate_measures = set()
        if calculate_measures is True:
            calculate_measures = {'iou_seg'}
        if save_predicted_segmentations is False:
            save_predicted_segmentations = set()
        if save_predicted_segmentations is True:
            save_predicted_segmentations = {'seg'}

        self._sequences_to_eval = sequences_to_eval
        self._save_predicted_segmentations = save_predicted_segmentations
        self._calculate_measures = calculate_measures
        self._debug_sequences = debug_sequences
        self._skip_existing = skip_existing

        assert calculate_measures <= {'iou_seg'}
        assert save_predicted_segmentations <= {'seg'}
        if 'seg' in save_predicted_segmentations:
            self._imsavehlp = utils.ImageSaveHelper() # Thread inside
            if dataset.__class__.__name__ == 'DAVIS17V2':
                self._sdm = utils.ReadSaveDAVISChallengeLabels()
            elif dataset.__class__.__name__ == 'YTVOSV2':
                self._sdm = utils.ReadSaveYTVOSChallengeLabels()
            else:
                raise NotImplementedError("Requested to save predicted segmentations with a dataset where this functionality is not supported. Dataset was {}".format(dataset.__class__.__name__))

    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                          for seganno in video_part['given_segannos']] 
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames

    def get_seg_measure(self, tracker_out, segannos, frame_idx0, seqname, obj_idx):
        B, N, C, H, W = tracker_out['segs'].size()
        if  N - frame_idx0 < 3:
            print("Object {} in sequence {} was tracked for only {} frames, should be at least 3 frames, setting iou to 1.".format(obj_idx, seqname, N - frame_idx0))
            measure = 1.0
        else:
            obj_segannos = (segannos[:,frame_idx0:,:,:] == obj_idx).long().view(N,H*W)
            obj_preds = (tracker_out['segs'][:,frame_idx0:,:,:,:] == obj_idx).long().view(N,H*W)
            intersection = torch.min(obj_segannos, obj_preds).sum(-1)
            union = torch.max(obj_segannos, obj_preds).sum(-1)
            measure = torch.clamp(intersection.float(), min=1e-10) / torch.clamp(union.float(), min=1e-10) # size N
        return measure

    def add_measures_for_chunk(self, measure_lsts, tracker_out, segannos, seqname):
        for obj_idx, first_frame_idx in tracker_out['object_visibility'].items():
            if measure_lsts.get(obj_idx) is None:
                measure_lsts[obj_idx] = []
            if 'iou_seg' in self._calculate_measures:
                measure = self.get_seg_measure(tracker_out, segannos, first_frame_idx, seqname, obj_idx)
            measure_lsts[obj_idx].append(measure)
        return measure_lsts

    def evaluate_video(self, model, seqname, video_parts, output_path):
        """
        """
        if self._save_predicted_segmentations:
            predicted_segmentation_lst = []
        if self._calculate_measures:
            measure_lsts = {}
#        tracker_out = None
        tracker_state = None
        for video_part in video_parts:
            images, given_segannos, segannos, fnames = self.read_video_part(video_part)

            try:
                tracker_out, tracker_state = model(images, given_segannos, tracker_state)
            except:
                print("Crash in model for seq {}, given segannos: {}".format(
                    seqname, [elem.size() if elem is not None else None for elem in given_segannos]))
                raise
            
            if 'seg' in self._save_predicted_segmentations:
                assert tracker_out['segs'].size(1) == len(fnames)
                for idx in range(len(fnames)):
                    fpath = os.path.join(output_path, seqname, fnames[idx])
                    data = ((tracker_out['segs'][0,idx,0,:,:].cpu().byte().numpy(), fpath), self._sdm)
                    self._imsavehlp.enqueue(data)
            if self._calculate_measures:
                measure_lsts = self.add_measures_for_chunk(measure_lsts, tracker_out, segannos, seqname)

#                record_file = os.path.join(
#                    self.result_dir, name, '%s.txt' % seq_name)
            
        if self._calculate_measures:
            measures = {}
            for obj_idx, measure_lst in measure_lsts.items():
                measures_tensor = torch.cat(measure_lst, dim=0)
                measures["{}_{:02d}".format(seqname, obj_idx)] = {
                    'mean': measures_tensor[1:-1].mean().item(),
                    'perframe': [elem.item() for elem in measures_tensor]
                }
                print(seqname, obj_idx, measures_tensor[1:-1].mean().item())
        else:
            measures = None
            print(seqname)

        if self._debug_sequences:
            raise NotImplementedError("Joakim probably played ping-pong instead of implementing this.")

        return measures
            
                    
    def evaluate(self, model, output_path):
        model.to(self._device)
        model.eval()
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        performance_measures = {'iou': {'mean':{}, 'perseq':{}}}

        t0 = time.time()
        eval_times = []
        with torch.no_grad():
            for seqname, video_parts in self._dataset.get_video_generator():
                if self._sequences_to_eval is not 'all' and seqname not in self._sequences_to_eval:
                    continue
                t1 = time.time()
                savepath = os.path.join(output_path, seqname)
                if os.path.exists(savepath) and len(os.listdir(savepath)) > 0 and self._skip_existing:
                    print("Sequence {} exists, skipping.".format(seqname))
                    continue
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                try:
                    measures = self.evaluate_video(model, seqname, video_parts, output_path)
                except:
                    print(seqname)
                    raise
                if self._calculate_measures:
                    performance_measures['iou']['perseq'].update(measures)
                eval_times.append(time.time() - t1)
                
        if self._calculate_measures:
            performance_measures['iou']['mean'] = (lambda lst: sum(lst)/len(lst))(
                [seq_measure['mean'] for seq_measure in performance_measures['iou']['perseq'].values()])
            result_path = output_path + "_results.json"
            with open(result_path, 'w') as fp:
                json.dump(performance_measures, fp)
            print("Storing result in {} with a mean iou of {:.3f}".format(
                output_path, performance_measures['iou']['mean']))

        if self._save_predicted_segmentations:
            self._imsavehlp.kill()


        
        
