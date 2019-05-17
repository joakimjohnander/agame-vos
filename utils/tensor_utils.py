import torch

def get_intersection_over_union(predictions, gt):
    """ Calculates the class intersections over unions of two tensors
    Args:
        predictions (Tensor): of size (nsamples,nclasses,H,W)
        gt (Tensor): Ground truth segmentations, of size
            (nsamples,H,W)
    Returns:
        Tensor: of size (nsamples,nclasses) with error for each class
    """
    nsamples,nclasses,height,width = predictions.size()
    assert gt.size(0) == nsamples, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(1) == height, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    assert gt.size(2) == width, "gt size: {},  predictions size: {}".format(gt.size(), predictions.size())
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = gt.new_tensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1) # [1,K,1,1]
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)    # [N,K,H,W]
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)                     # [N,K,H,W]
    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)            # [N,K]
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1)             # [N,K]
    assert (intersection > union).sum() == 0
    return (intersection + 1e-8) / (union + 1e-8)                                          # [N,K]
