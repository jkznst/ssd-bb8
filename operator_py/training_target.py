import mxnet as mx
import numpy as np



class TrainingTargets(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, overlap_threshold, negative_mining_ratio, negative_mining_thresh, variances):
        super(TrainingTargets, self).__init__()
        self.overlap_threshold = overlap_threshold
        self.negative_mining_ratio = negative_mining_ratio
        self.negative_mining_thresh = negative_mining_thresh
        self.variances = variances

        self.eps = 1e-14

    def forward(self, is_train, req, in_data, out_data, aux):

        anchors = in_data[0]    # 1 x num_anchors x 4
        class_preds = in_data[1]
        labels = in_data[2]

        box_target, box_mask, cls_target = mx.nd.contrib.MultiBoxTarget(anchors, labels, class_preds,
                                                                            overlap_threshold=self.overlap_threshold,
                                                                            ignore_label=-1,
                                                                            negative_mining_ratio=self.negative_mining_ratio,
                                                                            minimum_negative_samples=0,
                                                                            negative_mining_thresh=self.negative_mining_thresh,
                                                                            variances=self.variances,
                                                                            name="multibox_target")

        anchor_mask = box_mask.reshape(shape=(0, -1, 4))  # batchsize x num_anchors x 4
        bb8_mask = mx.nd.repeat(data=anchor_mask, repeats=4, axis=2)  # batchsize x num_anchors x 16
        # anchor_mask = mx.nd.mean(data=anchor_mask, axis=2, keepdims=False, exclude=False)

        anchors_in_use = mx.nd.broadcast_mul(lhs=anchor_mask, rhs=anchors)  # batchsize x num_anchors x 4

        # transform the anchors from [xmin, ymin, xmax, ymax] to [cx, cy, wx, hy]

        centerx = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1) +
                   mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3)) / 2
        centery = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2) +
                   mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4)) / 2
        width = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=2, end=3) -
                 mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=0, end=1)) + 1e-8
        height = (mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=3, end=4) -
                  mx.nd.slice_axis(data=anchors_in_use, axis=2, begin=1, end=2)) + 1e-8
        # anchors_in_use_transformed = mx.symbol.zeros_like(data=anchors_in_use)
        # anchors_in_use_transformed[:, :, 0] = (anchors_in_use[:, :, 0] + anchors_in_use[:, :, 2]) / 2
        # anchors_in_use_transformed[:, :, 1] = (anchors_in_use[:, :, 1] + anchors_in_use[:, :, 3]) / 2
        # anchors_in_use_transformed[:, :, 2] = anchors_in_use[:, :, 2] - anchors_in_use[:, :, 0] + 0.0000001
        # anchors_in_use_transformed[:, :, 3] = anchors_in_use[:, :, 3] - anchors_in_use[:, :, 1] + 0.0000001
        anchors_in_use_transformed = mx.nd.concat(centerx, centery, width, height, dim=2)

        # bb8_target = mx.symbol.zeros(shape=(32, 8732, 16))
        bb8_label = mx.nd.slice_axis(data=labels, axis=2, begin=8, end=24)

        # calculate targets for LINEMOD dataset
        bb8_target = mx.nd.slice_axis(data=bb8_label, axis=1, begin=0, end=1)
        # batchsize x 1 x 16, only consider the first label as gt
        # in order [x0,y0,x1,y1,...,x7,y7]

        # substract center
        bb8_target = mx.nd.broadcast_sub(bb8_target, mx.nd.tile(  # repeat single element !! error
            data=mx.nd.slice_axis(anchors_in_use_transformed, axis=2, begin=0, end=2),
            reps=(1, 1, 8)))
        # divide by w and h
        bb8_target = mx.nd.broadcast_div(bb8_target, mx.nd.tile(
            data=mx.nd.slice_axis(anchors_in_use_transformed, axis=2, begin=2, end=4),
            reps=(1, 1, 8))) / 0.05  # variance

        condition = bb8_mask > 0.5
        bb8_target = mx.nd.where(condition=condition, x=bb8_target, y=mx.nd.zeros_like(data=bb8_target))

        bb8_target = bb8_target.flatten()  # batchsize x (num_anchors x 16)
        bb8_mask = bb8_mask.flatten()  # batchsize x (num_anchors x 16)

        self.assign(out_data[0], req[0], box_target)
        self.assign(out_data[1], req[1], box_mask)
        self.assign(out_data[2], req[2], cls_target)
        self.assign(out_data[3], req[3], bb8_target)
        self.assign(out_data[4], req[4], bb8_mask)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("training_targets")
class TrainingTargetsProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, overlap_threshold=0.5, negative_mining_ratio=3,
                 negative_mining_thresh=0.5, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(TrainingTargetsProp, self).__init__(need_top_grad=False)
        self.overlap_threshold = float(overlap_threshold)
        self.negative_mining_ratio = float(negative_mining_ratio)
        self.negative_mining_thresh = float(negative_mining_thresh)
        self.variances = variances

    def list_arguments(self):
        return ['anchors', 'cls_preds', 'labels']

    def list_outputs(self):
        return ['box_target', 'box_mask', 'cls_target', 'bb8_target', 'bb8_mask']

    def infer_shape(self, in_shape):
        anchors_shape = in_shape[0]
        data_shape = in_shape[1]
        label_shape = in_shape[2]

        box_target_shape = (data_shape[0], 4 * data_shape[2])
        box_mask_shape = (data_shape[0], 4 * data_shape[2])
        cls_target_shape = (data_shape[0], data_shape[2])
        bb8_target_shape = (data_shape[0], 16 * data_shape[2])
        bb8_mask_shape = (data_shape[0], 16 * data_shape[2])

        return [anchors_shape, data_shape, label_shape], \
               [box_target_shape, box_mask_shape,
                cls_target_shape, bb8_target_shape, bb8_mask_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return TrainingTargets(self.overlap_threshold, self.negative_mining_ratio, self.negative_mining_thresh, self.variances)
