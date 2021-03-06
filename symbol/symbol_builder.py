import mxnet as mx
from symbol.common import multi_layer_feature, multibox_layer
# from operator_py.focal_loss import *
# from operator_py.focal_loss_layer import *
from operator_py.training_target import *


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)


# def training_targets(anchors, class_preds, labels, use_focalloss=False):
#     # labels_np = labels.asnumpy()
#     # view_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=6, end=7)
#     # inplane_cls_label = mx.nd.slice_axis(data=labels, axis=2, begin=7, end=8)
#     # bbox_label = mx.nd.slice_axis(data=labels, axis=2, begin=1, end=5)
#     # label_valid_count = mx.symbol.sum(mx.symbol.slice_axis(labels, axis=2, begin=0, end=1) >= 0, axis=1)
#     # class_preds = class_preds.transpose(axes=(0,2,1))
#     if use_focalloss:
#         box_target, box_mask, cls_target = mx.symbol.contrib.MultiBoxTarget(anchors, labels, class_preds,
#                                                                             overlap_threshold=.5,
#                                                                             ignore_label=-1, negative_mining_ratio=-1,
#                                                                             minimum_negative_samples=0,
#                                                                             negative_mining_thresh=.5,
#                                                                             variances=(0.1, 0.1, 0.2, 0.2),
#                                                                             name="multibox_target")
#     else:
#         box_target, box_mask, cls_target = mx.symbol.contrib.MultiBoxTarget(anchors, labels, class_preds, overlap_threshold=.5,
#             ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0,
#             negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
#             name="multibox_target")
#
#     anchor_mask = box_mask.reshape(shape=(0, -1, 4))    # batchsize x num_anchors x 4
#     bb8_mask = mx.symbol.repeat(data=anchor_mask, repeats=4, axis=2)  # batchsize x num_anchors x 16
#     #anchor_mask = mx.nd.mean(data=anchor_mask, axis=2, keepdims=False, exclude=False)
#
#     anchors_in_use = mx.symbol.broadcast_mul(lhs=anchor_mask,rhs=anchors)   # batchsize x num_anchors x 4
#
#     # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]
#
#     centerx = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) + \
#                mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
#     centery = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) + \
#                mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
#     width = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) - \
#                mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 0.0000001
#     height = (mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) - \
#                mx.symbol.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 0.0000001
#     # anchors_in_use_transformed = mx.symbol.zeros_like(data=anchors_in_use)
#     # anchors_in_use_transformed[:, :, 0] = (anchors_in_use[:, :, 0] + anchors_in_use[:, :, 2]) / 2
#     # anchors_in_use_transformed[:, :, 1] = (anchors_in_use[:, :, 1] + anchors_in_use[:, :, 3]) / 2
#     # anchors_in_use_transformed[:, :, 2] = anchors_in_use[:, :, 2] - anchors_in_use[:, :, 0] + 0.0000001
#     # anchors_in_use_transformed[:, :, 3] = anchors_in_use[:, :, 3] - anchors_in_use[:, :, 1] + 0.0000001
#     anchors_in_use_transformed = mx.symbol.concat(centerx, centery, width, height, dim=2)
#
#     # bb8_target = mx.symbol.zeros(shape=(32, 8732, 16))
#     bb8_label = mx.symbol.slice_axis(data=labels, axis=2, begin=8, end=24)
#     # calculate targets for OCCLUSION dataset
#     # for cid in range(1,9):
#     #     cid_target_mask = (cls_target == cid)
#     #     cid_target_mask = cid_target_mask.reshape(shape=(0,-1,1))
#     #     cid_anchors_in_use_transformed = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=anchors_in_use_transformed)
#     #     cid_label_mask = (mx.symbol.slice_axis(data=labels, axis=2, begin=0, end=1) == cid-1)
#     #     cid_bb8_label = mx.symbol.broadcast_mul(lhs=cid_label_mask, rhs=bb8_label)
#     #     cid_bb8_label = mx.symbol.sum(cid_bb8_label, axis=1, keepdims=True)
#     #     # substract center
#     #     cid_bb8_target = mx.symbol.broadcast_sub(cid_bb8_label, mx.symbol.tile(   # repeat single element !! error
#     #         data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=0, end=2),
#     #         reps=(1,1,8)))
#     #     # divide by w and h
#     #     cid_bb8_target = mx.symbol.broadcast_div(cid_bb8_target, mx.symbol.tile(
#     #         data=mx.symbol.slice_axis(cid_anchors_in_use_transformed, axis=2, begin=2, end=4),
#     #         reps=(1, 1, 8))) / 0.1  # variance
#     #     cid_bb8_target = mx.symbol.broadcast_mul(lhs=cid_target_mask, rhs=cid_bb8_target)
#     #     bb8_target = bb8_target + cid_bb8_target
#
#
#     # calculate targets for LINEMOD dataset
#     bb8_target = mx.symbol.slice_axis(data=bb8_label, axis=1, begin=0, end=1)    # batchsize x 1 x 16, only consider the first label as gt
#     # substract center
#     bb8_target = mx.symbol.broadcast_sub(bb8_target, mx.symbol.tile(   # repeat single element !! error
#         data=mx.symbol.slice_axis(anchors_in_use_transformed, axis=2, begin=0, end=2),
#         reps=(1,1,8)))
#     # divide by w and h
#     bb8_target = mx.symbol.broadcast_div(bb8_target, mx.symbol.tile(
#         data=mx.symbol.slice_axis(anchors_in_use_transformed, axis=2, begin=2, end=4),
#         reps=(1,1,8))) / 0.1 # variance
#
#     condition = bb8_mask > 0.5
#     bb8_target = mx.symbol.where(condition=condition, x=bb8_target, y=mx.symbol.zeros_like(data=bb8_target))
#
#     bb8_target = bb8_target.flatten()   # batchsize x (num_anchors x 16)
#     bb8_mask = bb8_mask.flatten()       # batchsize x (num_anchors x 16)
#     return box_target, box_mask, cls_target, bb8_target, bb8_mask


def get_symbol_train(network, num_classes, alpha_bb8, alpha_loc, use_dilated, use_focalloss, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, minimum_negative_samples=0, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections
    minimum_negative_samples : int
        always have some negative examples, no matter how many positive there are.
        this is useful when training on images with no ground-truth.
    Returns
    -------
    mx.Symbol

    """
    use_focalloss = False
    label = mx.sym.Variable('label')

    body = import_module(network).get_symbol(num_classes=num_classes, use_dilated=use_dilated, **kwargs)

    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # now cls_preds are in shape of  batchsize x num_class x num_anchors

    # loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = training_targets(anchors=anchor_boxes,
    #             class_preds=cls_preds, labels=label, use_focalloss=use_focalloss)

    loc_target, loc_target_mask, cls_target, bb8_target, bb8_target_mask = mx.sym.Custom(op_type="training_targets",
                                                                                         name="training_targets",
                                                                                         anchors=anchor_boxes,
                                                                                         cls_preds=cls_preds,
                                                                                         labels=label)

    # tmp = mx.contrib.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=minimum_negative_samples, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    # loc_target = tmp[0]
    # loc_target_mask = tmp[1]
    # cls_target = tmp[2]

    # if use_focalloss:
    # cls_prob_ = mx.sym.SoftmaxActivation(cls_preds, mode='channel')
    # cls_prob = mx.sym.Custom(cls_preds, cls_prob_, cls_target, op_type='focal_loss', name='cls_prob',
    #                          gamma=2.0, alpha=0.25, normalize=True)

    # cls_prob = mx.sym.Custom(op_type='FocalLoss', name='cls_prob', data=cls_preds, labels=cls_target, alpha=0.25, gamma=2)

    # else:
    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=alpha_loc, \
        normalization='valid', name="loc_loss")
    bb8_loss_ = mx.symbol.smooth_l1(name="bb8_loss_", \
        data=bb8_target_mask * (bb8_preds - bb8_target), scalar=1.0)
    bb8_loss = mx.symbol.MakeLoss(bb8_loss_, grad_scale=alpha_bb8, \
        normalization='valid', name="bb8_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    # anchor = mx.symbol.MakeLoss(data=mx.symbol.broadcast_mul(loc_target_mask.reshape((0,-1,4)), anchor_boxes), grad_scale=0, name='anchors')
    anchors = mx.symbol.MakeLoss(data=anchor_boxes, grad_scale=0, name='anchors')
    loc_mae = mx.symbol.MakeLoss(data=mx.sym.abs(loc_target_mask * (loc_preds - loc_target)),
                                 grad_scale=0, name='loc_mae')
    loc_label = mx.symbol.MakeLoss(data=loc_target_mask * loc_target, grad_scale=0., name='loc_label')
    loc_pred_masked = mx.symbol.MakeLoss(data=loc_target_mask * loc_preds, grad_scale=0, name='loc_pred_masked')
    bb8_label = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_target, grad_scale=0, name='bb8_label')
    bb8_pred = mx.symbol.MakeLoss(data=bb8_preds, grad_scale=0, name='bb8_pred')
    bb8_pred_masked = mx.symbol.MakeLoss(data=bb8_target_mask * bb8_preds, grad_scale=0, name='bb8_pred_masked')
    bb8_mae = mx.symbol.MakeLoss(data=mx.sym.abs(bb8_target_mask * (bb8_preds - bb8_target)),
                                 grad_scale=0, name='bb8_mae')

    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")
    loc_pred = mx.symbol.MakeLoss(data=loc_preds, grad_scale=0, name='loc_pred')

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae])
    return out

def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    body = import_module(network).get_symbol(num_classes=num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes, bb8_preds = multibox_layer(layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
