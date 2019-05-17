from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils

DEVICE=torch.device("cuda")

def softmax_aggregate(predicted_seg, object_ids):
    """
    args:
        predicted_seg (dict of N tensors (B x 2 x H x W)): unfused p(obj0), p(not obj0) and so on
        object_ids (list of object ids): In general, has the form [1,2,...]
    returns:
        tensor (B x 1 x H x W): contains output hard-maxed segmentation map
        dict (idx: tensor (B x 2 x H x W): contains p(obj0), p(not obj0), p(obj1), p(not obj1), ...
    """
    bg_seg, _ = torch.stack([seg[:,0,:,:] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:,1:,:,:].clamp(1e-7, 1 - 1e-7) / seg[:,0,:,:].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:,1::2,:,:].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx+1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:,2*(idx+1):2*(idx+2),:,:] for idx, obj_idx in enumerate(object_ids)}

def get_required_padding(height, width, div):
    height_pad = (div - height % div) % div
    width_pad = (div - width % div) % div
    padding = [(width_pad+1)//2, width_pad//2, (height_pad+1)//2, height_pad//2]
    return padding

def apply_padding(x, y, padding):
    B, L, C, H, W = x.size()
    x = x.view(B*L, C, H, W)
    x = F.pad(x, padding, mode='reflect')
    _, _, height, width = x.size()
    x = x.view(B, L, C, height, width)
    y = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in y]
    return x, y

def unpad(tensor, padding):
    if isinstance(tensor, (dict, OrderedDict)):
        return {key: unpad(val, padding) for key, val in tensor.items()}
    elif isinstance(tensor, (list, tuple)):
        return [unpad(elem, padding) for elem in tensor]
    else:
        _, _, _, height, width = tensor.size()
        tensor = tensor[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]
        return tensor

class LinearRelu(nn.Sequential):
    def __init__(self, *linear_args):
        super().__init__()
        self.add_module('linear', nn.Linear(*linear_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
class DilationpyramidRelu(nn.Module):
    def __init__(self, nchannels_in, nchannels_out, kernel_size, stride, paddings, dilations):
        super().__init__()
        assert len(paddings) == len(dilations)
        self.nlevels = len(paddings)
        for i in range(self.nlevels):
            self.add_module('conv{:d}'.format(i), nn.Conv2d(nchannels_in, nchannels_out, kernel_size, stride, paddings[i], dilations[i]))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    
    def forward(self, x):
        h = []
        for i in range(self.nlevels):
            h.append(getattr(self, 'conv{:d}'.format(i))(x))
        h = torch.cat(h, dim=-3)
        h = self.naf(h)
        return h

class GaussiansAgame(nn.Module):
    def __init__(self, nchannels_in, nchannels_lda, lr, cov_reg_init, residual=True, covest=True):
        super().__init__()
        self.lr = lr
        self.residual = residual
        self.covest = covest
        self.conv_in = nn.Conv2d(nchannels_in, nchannels_lda, 1)
        self.cov_reg = nn.Parameter(cov_reg_init * torch.ones(4, 1, nchannels_lda, 1, 1))
        nn.init.kaiming_uniform_(self.conv_in.weight)
    def get_init_state(self, ref_feats, ref_seg):
        B, C, H, W = ref_feats['s16'].size()
        N = H * W
        K1 = ref_seg.size(1)
        conv_ref_feats = self.conv_in(ref_feats['s16'])
        means = [F.adaptive_avg_pool2d(conv_ref_feats * ref_seg[:,i:i+1,:,:], 1)
                 / (1/N + F.adaptive_avg_pool2d(ref_seg[:,i:i+1,:,:], 1))
                 for i in range(K1)]
        distances = [(conv_ref_feats - means[i]) for i in range(K1)]
        if self.covest:
            covariances = [F.softplus(self.cov_reg[i]) + F.adaptive_avg_pool2d(distances[i]
                                                                               * distances[i]
                                                                               * ref_seg[:,i:i+1,:,:], 1)
                           / (1/N + F.adaptive_avg_pool2d(ref_seg[:,i:i+1,:,:], 1))
                           for i in range(K1)]
        else:
            covariances = [F.softplus(self.cov_reg[i]) for i in range(K1)]
        if self.residual:
            test_classification = F.softmax(self.forward(ref_feats, {'appmod': (means, covariances)})[0], dim=-3)
            residual = F.relu(test_classification[:,:,:,:] - ref_seg[:,:,:,:]).split(1, dim=1)
            means += [(F.adaptive_avg_pool2d(conv_ref_feats * residual[idx], 1)
                       / (1/N + F.adaptive_avg_pool2d(residual[idx], 1)))
                      for idx in range(K1)]
            distances += [(conv_ref_feats - means[i+2]) for i in range(K1)]
            if self.covest:
                covariances += [F.softplus(self.cov_reg[i+2])+F.adaptive_avg_pool2d(distances[i+2]
                                                                                    * distances[i+2]
                                                                                    * residual[i], 1)
                                / (1/N + F.adaptive_avg_pool2d(residual[i], 1))
                                for i in range(K1)]
            else:
                covariances += [F.softplus(self.cov_reg[i+2]) for i in range(K1)]
        return means, covariances
    def update(self, feats, seg, state):
        old_means, old_covariances = state['appmod']
        new_means, new_covariances = self.get_init_state(feats, seg)
        means = [(1 - self.lr) * o + self.lr * n for o,n in zip(old_means, new_means)]
        covariances = [(1 - self.lr) * o + self.lr * n for o,n in zip(old_covariances, new_covariances)]
        return means, covariances
    def forward(self, feats, state):
        means, covariances = state['appmod']
        nclasses = len(means)
        conv_feats = self.conv_in(feats['s16'])
        distances = [(conv_feats - means[i]) for i in range(nclasses)]
        scores = [- 1/2 * (torch.log(covariances[i])
                           + (distances[i] * distances[i]) / covariances[i]).sum(dim=-3,keepdim=True)
                  for i in range(nclasses)]
        scores_tensor = torch.cat(scores, dim=-3)
        return scores_tensor, (means, covariances)

class GaussiansAgameHack(nn.Module):
    def __init__(self, nchannels_in, nchannels_lda, lr, cov_reg_init, residual=True, covest=True,
                 logprob_bias=None, logprob_trainable_bias=None):
        super().__init__()
        self.lr = lr
        self.residual = residual
        self.covest = covest
        self.conv_in = nn.Conv2d(nchannels_in, nchannels_lda, 1)
        self.cov_reg = nn.Parameter(cov_reg_init * torch.ones(4, 1, nchannels_lda, 1, 1))
        self.logprob_bias = logprob_bias
        if logprob_trainable_bias is not None:
            self.logprob_trainable_bias = nn.Parameter(logprob_trainable_bias * torch.ones(4))
        else:
            self.logprob_trainable_bias = None
        nn.init.kaiming_uniform_(self.conv_in.weight)
    def get_init_state(self, ref_feats, ref_seg):
        B, C, H, W = ref_feats['s16'].size()
        N = H * W
        K1 = ref_seg.size(1)
        conv_ref_feats = self.conv_in(ref_feats['s16'])
        means = [F.adaptive_avg_pool2d(conv_ref_feats * ref_seg[:,i:i+1,:,:], 1)
                 / (1/N + F.adaptive_avg_pool2d(ref_seg[:,i:i+1,:,:], 1))
                 for i in range(K1)]
        distances = [(conv_ref_feats - means[i]) for i in range(K1)]
        if self.covest:
            covariances = [F.softplus(self.cov_reg[i]) + F.adaptive_avg_pool2d(distances[i]
                                                                               * distances[i]
                                                                               * ref_seg[:,i:i+1,:,:], 1)
                           / (1/N + F.adaptive_avg_pool2d(ref_seg[:,i:i+1,:,:], 1))
                           for i in range(K1)]
        else:
            covariances = [F.softplus(self.cov_reg[i]) for i in range(K1)]
        if self.residual:
            test_classification = F.softmax(self.forward(ref_feats, {'appmod': (means, covariances)})[0], dim=-3)
            residual = F.relu(test_classification[:,:,:,:] - ref_seg[:,:,:,:]).split(1, dim=1)
            means += [(F.adaptive_avg_pool2d(conv_ref_feats * residual[idx], 1)
                       / (1/N + F.adaptive_avg_pool2d(residual[idx], 1)))
                      for idx in range(K1)]
            distances += [(conv_ref_feats - means[i+2]) for i in range(K1)]
            if self.covest:
                covariances += [F.softplus(self.cov_reg[i+2])+F.adaptive_avg_pool2d(distances[i+2]
                                                                                    * distances[i+2]
                                                                                    * residual[i], 1)
                                / (1/N + F.adaptive_avg_pool2d(residual[i], 1))
                                for i in range(K1)]
            else:
                covariances += [F.softplus(self.cov_reg[i+2]) for i in range(K1)]
        return means, covariances
    def update(self, feats, seg, state):
        old_means, old_covariances = state['appmod']
        new_means, new_covariances = self.get_init_state(feats, seg)
        means = [(1 - self.lr) * o + self.lr * n for o,n in zip(old_means, new_means)]
        covariances = [(1 - self.lr) * o + self.lr * n for o,n in zip(old_covariances, new_covariances)]
        return means, covariances
    def forward(self, feats, state):
        means, covariances = state['appmod']
        nclasses = len(means)
        conv_feats = self.conv_in(feats['s16'])
        distances = [(conv_feats - means[i]) for i in range(nclasses)]
        scores = [- 1/2 * (torch.log(covariances[i])
                           + (distances[i] * distances[i]) / covariances[i]).sum(dim=-3,keepdim=True)
                  for i in range(nclasses)]
        if self.logprob_bias is not None:
            scores = [scores[i] + self.logprob_bias[i] for i in range(nclasses)]
        if self.logprob_trainable_bias is not None:
            scores = [scores[i] + self.logprob_trainable_bias[i] for i in range(nclasses)]
        scores_tensor = torch.cat(scores, dim=-3)
        return scores_tensor, (means, covariances)
    
class RGMPLike(nn.Module):
    def __init__(self, children_cfg):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('layer{:02d}'.format(idx), SUBMODS[child_cfg[0]](*child_cfg[1]))
            for idx, child_cfg in enumerate(children_cfg)]))
    def get_init_state(self, input_feats, given_seg):
        dynmod_state = {'init_feats': input_feats['s16'], 'init_seg': given_seg, 'prev_seg': given_seg}
        return dynmod_state
    def update(self, input_feats, predicted_segmentation, state):
        dynmod_state = state['dynmod']
        dynmod_state['prev_seg'] = predicted_segmentation
        return dynmod_state
    def forward(self, input_feats, state):
        dynmod_state = state['dynmod']
        h = torch.cat([input_feats['s16'], dynmod_state['init_feats'],
                       dynmod_state['prev_seg'], dynmod_state['init_seg']], dim=-3)
        h = self.layers(h)
        return h, dynmod_state
    
class RGMPLikeNoInit(nn.Module):
    def __init__(self, children_cfg):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('layer{:02d}'.format(idx), SUBMODS[child_cfg[0]](*child_cfg[1]))
            for idx, child_cfg in enumerate(children_cfg)]))
    def get_init_state(self, input_feats, given_seg):
        dynmod_state = {'prev_seg': given_seg}
        return dynmod_state
    def update(self, input_feats, predicted_segmentation, state):
        dynmod_state = state['dynmod']
        dynmod_state['prev_seg'] = predicted_segmentation
        return dynmod_state
    def forward(self, input_feats, state):
        dynmod_state = state['dynmod']
        h = torch.cat([input_feats['s16'], dynmod_state['prev_seg']], dim=-3)
        h = self.layers(h)
        return h, dynmod_state

class FusionAgame(nn.Module): # Feed Forward Stack
    def __init__(self, layers_cfg, predictor_cfg):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('layer{:02d}'.format(idx), SUBMODS[child_cfg[0]](*child_cfg[1]))
            for idx, child_cfg in enumerate(layers_cfg)]))
        self.predictor = nn.Sequential(OrderedDict([
            ('layer{:02d}'.format(idx), SUBMODS[child_cfg[0]](*child_cfg[1]))
            for idx, child_cfg in enumerate(predictor_cfg)]))
    def forward(self, app_feats, dyn_feats, state):
        h = torch.cat([app_feats, dyn_feats], dim=-3)
        h = self.layers(h)
        coarse_segscore = self.predictor(h)
        return h, coarse_segscore
        
class UpsampleAgame(nn.Module):
    """Based on Piotr Dollar's sharpmask, fairly standard U-net/ladderstyle like upsampling path
    """
    def __init__(self, nchannels_feats, nchannels):
        super().__init__()
        self.project_s8 = SUBMODS['ConvRelu'](nchannels_feats['s8'], nchannels, 1, 1, 0)
        self.blend_s8 = SUBMODS['ConvRelu'](nchannels*2, nchannels, 3, 1, 1)
        self.project_s4 = SUBMODS['ConvRelu'](nchannels_feats['s4'], nchannels, 1, 1, 0)
        self.blend_s4 = SUBMODS['ConvRelu'](nchannels*2, nchannels, 3, 1, 1)
        self.predictor = SUBMODS['Conv'](nchannels, 2, 3, 1, 1)
    def forward(self, feats, fused_feats, state):
        h = fused_feats
        h = torch.cat([self.project_s8(feats['s8']), F.upsample(h, scale_factor=2, mode='bilinear')], dim=-3)
        h = self.blend_s8(h)
        h = torch.cat([self.project_s4(feats['s4']), F.upsample(h, scale_factor=2, mode='bilinear')], dim=-3)
        h = self.blend_s4(h)
        h = self.predictor(h)
        h = F.upsample(h, scale_factor=4, mode='bilinear')
        return h

APPMODS = {'GaussiansAgame': GaussiansAgame, 'GaussiansAgameHack': GaussiansAgameHack}
DYNMODS = {'RGMPLike': RGMPLike, 'RGMPLikeNoInit': RGMPLikeNoInit}
FUSMODS = {'FusionAgame': FusionAgame}
SEGMODS = {'UpsampleAgame': UpsampleAgame}
SUBMODS = {'DilationpyramidRelu': DilationpyramidRelu, 'ConvRelu': ConvRelu, 'Conv': Conv, 'LinearRelu': LinearRelu}

class TrackSeg(nn.Module):
    def __init__(self, backbone_cfg, appmod_cfg, dynmod_cfg, fusmod_cfg, segmod_cfg):
        super().__init__()
        self.backbone = getattr(models.backbones, backbone_cfg[0])(*backbone_cfg[1])
        self.appmod   = APPMODS[appmod_cfg[0]](*appmod_cfg[1])
        self.dynmod   = DYNMODS[dynmod_cfg[0]](*dynmod_cfg[1])
        self.fusmod   = FUSMODS[fusmod_cfg[0]](*fusmod_cfg[1])
        self.segmod   = SEGMODS[segmod_cfg[0]](*segmod_cfg[1])
        
    def get_init_state(self, image, given_seg):
        feats = self.backbone.get_features(image)
        state = {}
        state['appmod'] = self.appmod.get_init_state(feats, given_seg)
        state['dynmod'] = self.dynmod.get_init_state(feats, given_seg)
        return state

    def update(self, feats, seg, state):
        state['appmod'] = self.appmod.update(feats, seg, state)
        state['dynmod'] = self.dynmod.update(feats, seg, state)
        return state

    def extract_feats(self, img):
        feats = self.backbone.get_features(img)
        return feats
    
    def forward(self, feats, state):
        appmod_output, state['appmod'] = self.appmod(feats, state)
        dynmod_output, state['dynmod'] = self.dynmod(feats, state)
        fused_output, coarse_segscore = self.fusmod(appmod_output, dynmod_output, state)

        segscore = self.segmod(feats, fused_output, state)
        return state, coarse_segscore, segscore
        
class AGAME(nn.Module):
    def __init__(self,
                 backbone=('resnet101s16', (True, ('layer4',),('layer4',),('layer2',),('layer1',))),
                 appearance=('GaussiansAgame', (2048, 512, .1, -2)),
                 dynamic=('RGMPLike', ([('ConvRelu', (2*2048+2*2,512,1,1,0)),
                                        ('DilationpyramidRelu',(512,512,3,1,(1,3,6),(1,3,6))),
                                        ('ConvRelu',(1536,512,3,1,1))],)),
                 fusion=('FusionAgame',
                         ([('ConvRelu',(516,512,3,1,1)), ('ConvRelu',(512,128,3,1,1))],
                          [('Conv', (128, 2, 3, 1, 1))])),
                 segmod=('UpsampleAgame', ({'s8':512,'s4':256}, 128)),
                 update_with_fine_scores=False, update_with_softmax_aggregation=False, process_first_frame=True,
                 output_logsegs=True, output_coarse_logsegs=True, output_segs=True):
        super().__init__()
        self.backbone   = backbone
        self.appearance = appearance
        self.dynamic    = dynamic
        self.fusion     = fusion
        self.segmod     = segmod

        self.trackseg = TrackSeg(backbone, appearance, dynamic, fusion, segmod)

        self.update_with_fine_scores         = update_with_fine_scores
        self.update_with_softmax_aggregation = update_with_softmax_aggregation
        self.process_first_frame             = process_first_frame
        self.output_logsegs                  = output_logsegs
        self.output_coarse_logsegs           = output_coarse_logsegs
        self.output_segs                     = output_segs

    def forward(self, x, given_labels=None, state=None):
        """ Please note that with multiple targets, we loop over targets at this level
        Dimensionality abbreviations are:
            B: batch size
            L: temporal length, or sequence length
            C: number of channels
            H: height of image or feature map
            W: width of image or feature map
        params:
            x (Tensor): Video data of size (B,L,C,H,W)
            given_labels (List): Initial segmentations, as a list of (None OR (B,1,H,W) tensor)
            state (dict): Contains entire state at a given time step
        returns:
            Dict of tensors:
                logsegs (dict of Tensor): one element for each tracked object, each tensor of size (B,L,2,H,W)
                coarse_logsegs (dict of Tensor): one element for each tracked object, each tensor of size (B,L,2,H,W)
                segs (Tensor): Of size (B,L,1,H,W)
        """
        batchsize, nframes, nchannels, prepad_height, prepad_width = x.size()
        if given_labels is None:
            given_labels = nframes*[None]
        if given_labels is not None and not isinstance(given_labels, (tuple, list)):
            given_labels = [given_labels] + (nframes - 1)*[None]

        # Pad the input image, useful during evaluation where sequences can have any size
        # The padded part is removed in the end of the forward pass
        required_padding = get_required_padding(prepad_height, prepad_width, 16)
        if tuple(required_padding) != (0,0,0,0):
            assert not self.training, "Images should not be padded during training"
            x, given_labels = apply_padding(x, given_labels, required_padding)
        _, _, _, height, width = x.size()

        video_frames = [elem.view(batchsize, nchannels, height, width) for elem in x.split(1, dim=1)]

        # Construct state if not given, during initialization for instance
        if state is None:
            init_label = given_labels[0]
            assert init_label is not None, "Model needs either a state, or info that permits its initialization"

            object_ids = init_label.unique().tolist()
            if 0 in object_ids: object_ids.remove(0) # Background is not an object
            assert len(object_ids) > 0, "There are no objects given to track in the first frame"

            state = {}
            for obj_idx in object_ids:
                given_seg = F.avg_pool2d(torch.cat([init_label!=obj_idx, init_label==obj_idx], dim=-3).float(), 16)
                state[obj_idx] = self.trackseg.get_init_state(video_frames[0], given_seg)
        else: # use previous state, update labels if needed
            object_ids = list(state.keys())
            init_label = given_labels[0] if isinstance(given_labels, (tuple, list)) else given_labels
            if init_label is not None:
                new_object_ids = init_label.unique().tolist()
                if 0 in object_ids: object_ids.remove(0)
                
                for obj_idx in new_object_ids:
                    given_seg = F.avg_pool2d(torch.cat([init_label!=obj_idx, init_label==obj_idx], dim=-3).float(), 16)
                    if state.get(obj_idx) is None:
                        state[obj_idx] = self.trackseg.get_init_state(video_frames[0], given_seg)
                    else:
                        raise NotImplementedError("Received a given (ground-truth) segmentation for an object idx that is already initialized. This could happen in the future, but should not happen with standard VOS datasets. Existing ids are {} and new ids are {}".format(state.keys(), new_object_ids))
                object_ids = object_ids + new_object_ids
        object_visibility = {obj_idx: 0 for obj_idx in object_ids}

        logseg_lsts = {k: [] for k in object_ids}            # Used for training
        coarse_logseg_lsts = {k: [] for k in object_ids}     # Used for training
        seg_lst = []                                         # Used during inference
        if self.process_first_frame or given_labels[0] is None: # If no input_seg, process
            frames_to_process = range(0, nframes)
        else:
            # Use given mask when available, regularize it to deal with log(0) (we always output log-probs)
            for k in object_ids:
                if self.output_logsegs:
                    tmp = (given_labels[0] == k).float().clamp(1e-7, 1 - 1e-7)
                    tmp = torch.cat([1 - tmp, tmp], dim=1)
                    logseg_lsts[k].append(tmp.log())
                if self.output_coarse_logsegs:
                    tmp = (given_labels[0] == k).float().clamp(1e-7, 1 - 1e-7)
                    tmp = torch.cat([1 - tmp, tmp], dim=1)
                    coarse_logseg_lsts[k].append(tmp.log())
            if self.output_segs:
                seg_lst.append(given_labels[0])

            frames_to_process = range(1, nframes)

        # Iterate over frames
        for i in frames_to_process:
            # Feature extraction done outside since merging is done outside and we want to avoid mult. feat. ext.
            feats = self.trackseg.extract_feats(video_frames[i])

            # See if there is any given label, that is, if more targets should be tracked
            if isinstance(given_labels, (list, tuple)) and given_labels[i] is not None and i != 0:
                new_object_ids = given_labels[i].unique().tolist()
                if 0 in new_object_ids: new_object_ids.remove(0)
                
                for obj_idx in new_object_ids:
                    given_label_as_segmap_lst = [given_labels[i]!=obj_idx, given_labels[i]==obj_idx]
                    given_seg = F.avg_pool2d(torch.cat(given_label_as_segmap_lst, dim=-3).float(), 16)
                    if state.get(obj_idx) is None:
                        state[obj_idx] = self.trackseg.get_init_state(video_frames[i], given_seg)
                        object_visibility[obj_idx] = i
                    else:
                        raise NotImplementedError("Received a given (ground-truth) segmentation for an object idx that is already initialized. This could happen in the future, but should not happen with standard VOS datasets. Existing ids are {} and new ids are {}".format(state.keys(), new_object_ids))
                object_ids = object_ids + new_object_ids
            
            # Infer one time-step through model
            coarse_segscore = {}
            segscore = {}
            for k in object_ids:
                state[k], coarse_segscore[k], segscore[k] = self.trackseg(feats, state[k])

            # Merge scores for different objects (if necessary)
            predicted_seg = {k: F.softmax(segscore[k], dim=-3) for k in object_ids} # {objidx: B x 2 x H x W}
            if self.output_segs or self.update_with_softmax_aggregation:
                assert len(predicted_seg.values()) > 0, "i: {}, objids: {}".format(i, object_ids)
                output_seg, aggregated_seg = softmax_aggregate(predicted_seg, object_ids)

            if self.update_with_softmax_aggregation:
                assert not self.training, "Cannot update state with softmax-aggregated scores during training (yet)"
                update_seg = {n: F.avg_pool2d(aggregated_seg[n], 16) for n in object_ids}
            elif self.update_with_fine_scores:
                assert not self.training, "Cannot update state with fine scores during training, long recurrency..."
                update_seg = {k: F.avg_pool2d(predicted_seg[k], 16) for k in object_ids}
            else:
                update_seg = {k: F.softmax(coarse_segscore[k], dim=-3) for k in object_ids}

            # Subsequent model update (needed by appearance module, and mask-propagation)
            for k in object_ids:
                state[k] = self.trackseg.update(feats, update_seg[k], state[k])

            # Construct output
            for k in object_ids:
                if self.output_logsegs:
                    logseg_lsts[k].append(F.log_softmax(segscore[k], dim=-3))
                if self.output_coarse_logsegs:
                    coarse_logseg_lsts[k].append(F.interpolate(F.log_softmax(coarse_segscore[k], dim=-3),
                                                               scale_factor=16, mode='bilinear'))
            if self.output_segs:
                if isinstance(given_labels, (list, tuple)) and given_labels[i] is not None and i == 0:
                    seg_lst.append(given_labels[i])
                else:
                    seg_lst.append(output_seg)

        # Concatenate output from different frames into contiguous tensor
        output = {}
        output['object_visibility'] = object_visibility
        if self.output_logsegs:
            output['logsegs'] = {k: torch.stack(logseg_lsts[k], dim=1) for k in object_ids}
        if self.output_coarse_logsegs:
            output['coarse_logsegs'] = {k: torch.stack(coarse_logseg_lsts[k], dim=1) for k in object_ids}
        if self.output_segs:
            output['segs'] = torch.stack(seg_lst, dim=1)

        # Remove the padding from all output parts
        if tuple(required_padding) != (0,0,0,0):
            assert not self.training, "Padding is not intended during training"
            if self.output_logsegs:
                output['logsegs'] = unpad(output['logsegs'], required_padding)
            if self.output_coarse_logsegs:
                output['coarse_logsegs'] = unpad(output['coarse_logsegs'], required_padding)
            if self.output_segs:
                output['segs'] = unpad(output['segs'], required_padding)

#        utils.print_memory()
        return output, state


