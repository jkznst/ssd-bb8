import mxnet as mx
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
if sys.platform == "darwin":
    import ruamel.yaml as yaml
else:
    import ruamel.yaml as yaml

from MultiBoxDetection import BB8MultiBoxDetection
from EPnP.EPnP import EPnP
from obj_pose_eval import inout, pose_error, renderer, transform


def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f, Loader=yaml.CLoader)
        return content


class MApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    ovp_thresh : float
        overlap threshold for TP
    use_difficult : boolean
        use difficult ground-truths if applicable, otherwise just ignore
    class_names : list of str
        optional, if provided, will print out AP for each class
    pred_idx : int
        prediction index in network output list
    roc_output_path
        optional, if provided, will save a ROC graph for each class
    tensorboard_path
        optional, if provided, will save a ROC graph to tensorboard
    """
    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None,
                 pred_idx=0, roc_output_path=None, tensorboard_path=None):
        super(MApMetric, self).__init__('mAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = class_names + ['mAP']
            self.num = num + 1
        self.reset()
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)
        self.roc_output_path = roc_output_path
        self.tensorboard_path = tensorboard_path

    def save_roc_graph(self, recall=None, prec=None, classkey=1, path=None, ap=None):
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path = os.path.join(path, 'roc_'+self.class_names[classkey])
        if os.path.exists(plot_path):
            os.remove(plot_path)
        fig = plt.figure()
        plt.title(self.class_names[classkey])
        plt.plot(recall, prec, 'b', label='AP = %0.2f' % ap)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(plot_path)
        plt.close(fig)

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self.records = dict()
        self.counts = dict()

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def update(self, labels, preds):
        """
        Update internal records. This function now only update internal buffer,
        sum_metric and num_inst are updated in _update() function instead when
        get() is called to return results.

        Params:
        ----------
        labels: mx.nd.array (n * 6) or (n * 5), difficult column is optional
            2-d array of ground-truths, n objects(id-xmin-ymin-xmax-ymax-[difficult])
        preds: [cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae]
        """
        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        # get generated multi label from network
        cls_prob = preds[0]
        loc_pred = preds[4]
        bb8_pred = preds[5]
        anchors = preds[6]
        # anchor_in_use = anchors[anchors.nonzero()]
        bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
                                       variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        # independant execution for each image
        for i in range(labels[0].shape[0]):
            # get as numpy arrays
            label = labels[0][i].asnumpy()
            pred = bb8dets[i].asnumpy()
            # calculate for each class
            while (pred.shape[0] > 0):
                cid = int(pred[0, 0])
                indices = np.where(pred[:, 0].astype(int) == cid)[0]
                if cid < 0:
                    pred = np.delete(pred, indices, axis=0)
                    continue
                dets = pred[indices]
                pred = np.delete(pred, indices, axis=0)
                # sort by score, desceding
                # dets[dets[:,1].argsort()[::-1]]
                records = np.hstack((dets[:, 1][:, np.newaxis], np.zeros((dets.shape[0], 1))))
                # ground-truths
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                gts = label[label_indices, :]
                label = np.delete(label, label_indices, axis=0)
                if gts.size > 0:
                    found = [False] * gts.shape[0]
                    for j in range(dets.shape[0]):
                        # compute overlaps
                        ious = iou(dets[j, 2:6], gts[:, 1:5])
                        ovargmax = np.argmax(ious)
                        ovmax = ious[ovargmax]
                        if ovmax > self.ovp_thresh:
                            if (not self.use_difficult and
                                gts.shape[1] >= 6 and
                                gts[ovargmax, 5] > 0):
                                pass
                            else:
                                if not found[ovargmax]:
                                    records[j, -1] = 1  # tp
                                    found[ovargmax] = True
                                else:
                                    # duplicate
                                    records[j, -1] = 2  # fp
                        else:
                            records[j, -1] = 2 # fp
                else:
                    # no gt, mark all fp
                    records[:, -1] = 2

                # ground truth count
                if (not self.use_difficult and gts.shape[1] >= 6):
                    gt_count = np.sum(gts[:, 5] < 1)
                else:
                    gt_count = gts.shape[0]

                # now we push records to buffer
                # first column: score, second column: tp/fp
                # 0: not set(matched to difficult or something), 1: tp, 2: fp
                records = records[np.where(records[:, -1] > 0)[0], :]
                if records.size > 0:
                    self._insert(cid, records, gt_count)

            # add missing class if not present in prediction
            while (label.shape[0] > 0):
                cid = int(label[0, 0])
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                label = np.delete(label, label_indices, axis=0)
                if cid < 0:
                    continue
                gt_count = label_indices.size
                self._insert(cid, np.array([[0, 0]]), gt_count)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        for k, v in self.records.items():
            recall, prec = self._recall_prec(v, self.counts[k])
            ap = self._average_precision(recall, prec)
            if self.roc_output_path is not None:
                self.save_roc_graph(recall=recall, prec=prec, classkey=k, path=self.roc_output_path, ap=ap)
            aps.append(ap)
            if self.num is not None and k < (self.num - 1):
                self.sum_metric[k] = ap
                self.num_inst[k] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.mean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.mean(aps)

    def _recall_prec(self, record, count):
        """ get recall and precision from internal records """
        record = np.delete(record, np.where(record[:, 1].astype(int) == 0)[0], axis=0)
        sorted_records = record[record[:,0].argsort()[::-1]]
        tp = np.cumsum(sorted_records[:, 1].astype(int) == 1)
        fp = np.cumsum(sorted_records[:, 1].astype(int) == 2)
        if count <= 0:
            recall = tp * 0.0
        else:
            recall = tp / float(count)
        prec = tp.astype(float) / (tp + fp)
        return recall, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _insert(self, key, records, count):
        """ Insert records according to key """
        if key not in self.records:
            assert key not in self.counts
            self.records[key] = records
            self.counts[key] = count
        else:
            self.records[key] = np.vstack((self.records[key], records))
            assert key in self.counts
            self.counts[key] += count


class VOC07MApMetric(MApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset """
    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap


class PoseMetric(mx.metric.EvalMetric):
    """Calculate metrics for 6D pose estimation """
    def __init__(self, classes, LINEMOD_path,

                 eps=1e-8):
        super(PoseMetric, self).__init__('Pose')
        self.eps = eps
        self.num = 16
        self.ovp_thresh = 0.5
        self.use_difficult = False
        self.name = ['CrossEntropy', 'loc_SmoothL1', 'loc_MAE', 'loc_MAE_pixel', 'bb8_SmoothL1',
                     'bb8_MAE', 'bb8_MAE_pixel', 'ReprojectionError', 'ADD0.1', 'ADD0.3', 'ADD0.5',
                     're', 'te', 're&te', 'IoU2D0.5', 'IoU2D0.9']

        self.scale_to_meters = 0.001
        self.cam_intrinsic = np.zeros(shape=(3,4))
        cam_info = load_yaml(os.path.join(LINEMOD_path, 'camera.yml'))
        self.cam_intrinsic[0, 0] = cam_info['fx']
        self.cam_intrinsic[0, 2] = cam_info['cx']
        self.cam_intrinsic[1, 1] = cam_info['fy']
        self.cam_intrinsic[1, 2] = cam_info['cy']
        self.cam_intrinsic[2, 2] = 1

        models_info_path = os.path.join(LINEMOD_path, 'models/models_info.yml')
        self.models_info = load_yaml(models_info_path)
        self.classes = classes

        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        :param preds: [cls_prob, loc_loss, cls_label, bb8_loss, loc_pred, bb8_pred,
                           anchors, loc_label, loc_pred_masked, loc_mae, bb8_label, bb8_pred_masked, bb8_mae]
        Implementation of updating metrics
        """
        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        labels = labels[0].asnumpy()
        # get generated multi label from network
        cls_prob = preds[0]
        loc_loss = preds[1].asnumpy()     # smoothL1 loss
        loc_loss_in_use = loc_loss[loc_loss.nonzero()]
        cls_label = preds[2].asnumpy()
        bb8_loss = preds[3].asnumpy()
        loc_pred = preds[4]
        bb8_pred = preds[5]
        anchors = preds[6]
        # anchor_in_use = anchors[anchors.nonzero()]
        bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
                                      variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        bb8dets = bb8dets.asnumpy()

        # for i in range(1,16):
        #     if self.classes[0] == 'obj_{:02d}'.format(i):
        #         model_id = i
        #         break
        model_id = int(self.classes[0].strip('obj_'))

        for nbatch in range(bb8dets.shape[0]):

            self.num_inst[7] += 1
            self.num_inst[8] += 1
            self.num_inst[9] += 1
            self.num_inst[10] += 1
            self.num_inst[11] += 1
            self.num_inst[12] += 1
            self.num_inst[13] += 1
            self.num_inst[14] += 1
            self.num_inst[15] += 1

            if bb8dets[nbatch, 0, 0] == -1:
                continue
            else:
                # for LINEMOD dataset, for each image only select the first det
                # self.validate6Dpose(gt_pose=labels[nbatch, 0, 24:40], instance_bb8det=bb8dets[nbatch, 0, :], model_id=model_id)
                pose_est = self.calculate6Dpose(instance_bb8det=bb8dets[nbatch, 0, :], model_id=model_id)
                model_path = '/data/ZHANGXIN/DATASETS/SIXD_CHALLENGE/LINEMOD/models/' + self.classes[int(bb8dets[nbatch, 0, 0])] + '.ply'
                model_ply = inout.load_ply(model_path)
                model_ply['pts'] = model_ply['pts'] * self.scale_to_meters
                pose_gt_transform = np.reshape(labels[nbatch, 0, 24:40], newshape=(4, 4))
                pose_gt = {"R": pose_gt_transform[0:3, 0:3],
                "t": pose_gt_transform[0:3, 3:4]}

                # absolute pose error
                rot_error = pose_error.re(R_est=pose_est["R"], R_gt=pose_gt["R"]) / np.pi * 180.
                trans_error = pose_error.te(t_est=pose_est["t"], t_gt=pose_gt["t"]) / 0.01

                # other pose metrics
                if model_id in [10, 11]:
                    add_metric = pose_error.adi(pose_est=pose_est, pose_gt=pose_gt,
                                                model=model_ply)  # use adi when object is eggbox or glue
                else:
                    add_metric = pose_error.add(pose_est=pose_est, pose_gt=pose_gt, model=model_ply)    # use adi when object is eggbox or glue
                reproj_metric = pose_error.reprojectionError(pose_est=pose_est, pose_gt=pose_gt,
                                                             model=model_ply, K=self.cam_intrinsic[:, 0:3])
                cou_metric = pose_error.cou(pose_est=pose_est, pose_gt=pose_gt,
                                            model=model_ply, im_size=(640, 480), K=self.cam_intrinsic[:, 0:3])

                # metric update
                if reproj_metric <= 5:  # reprojection error less than 5 pixels
                    self.sum_metric[7] += 1

                if add_metric <= self.models_info[bb8dets[nbatch, 0, 0] + model_id]['diameter'] * self.scale_to_meters * 0.1:   # ADD metric less than 0.1 * diameter
                    self.sum_metric[8] += 1

                if add_metric <= self.models_info[bb8dets[nbatch, 0, 0] + model_id]['diameter'] * self.scale_to_meters * 0.3:   # ADD metric less than 0.1 * diameter
                    self.sum_metric[9] += 1

                if add_metric <= self.models_info[bb8dets[nbatch, 0, 0] + model_id]['diameter'] * self.scale_to_meters * 0.5:   # ADD metric less than 0.1 * diameter
                    self.sum_metric[10] += 1

                if rot_error < 5:   # 5 degrees
                    self.sum_metric[11] += 1

                if trans_error < 5: # 5 cm
                    self.sum_metric[12] += 1

                if (rot_error < 5) and (trans_error < 5):   # 5 degrees and 5 cm
                    self.sum_metric[13] += 1

                if cou_metric < 0.5:    # 2D IoU greater than 0.5
                    self.sum_metric[14] += 1

                if cou_metric < 0.1:    # 2D IoU larger than 0.9
                    self.sum_metric[15] += 1



        loc_label = preds[7].asnumpy()
        loc_label_in_use = loc_label[loc_label.nonzero()]
        loc_pred_masked = preds[8].asnumpy()
        loc_pred_in_use = loc_pred_masked[loc_pred_masked.nonzero()]
        loc_mae = preds[9].asnumpy()
        loc_mae_in_use = loc_mae[loc_mae.nonzero()]
        loc_mae_pixel = np.abs((bb8dets[:, 0, 2:6] - labels[:, 0, 1:5]) * 300)   # need to be refined

        bb8_label = preds[10].asnumpy()
        bb8_label_in_use = bb8_label[bb8_label.nonzero()]
        bb8_pred = preds[11].asnumpy()
        bb8_pred_in_use = bb8_pred[bb8_pred.nonzero()]
        bb8_mae = preds[12].asnumpy()
        bb8_mae_in_use = bb8_mae[bb8_mae.nonzero()]
        bb8_mae_pixel = np.abs((labels[:, 0, 8:24] - bb8dets[:, 0, 6:22]) * 300)  # need to be refined
        bb8_mae_pixel_x = bb8_mae_pixel[:, [0,2,4,6,8,10,12,14]]
        bb8_mae_pixel_y = bb8_mae_pixel[:, [1,3,5,7,9,11,13,15]]
        bb8_mae_pixel = np.sqrt(np.square(bb8_mae_pixel_x) + np.square(bb8_mae_pixel_y))
        # multi objects in one image (to be done)
        # loc_mae_pixel = []
        # bb8_mae_pixel = []
        # # independant execution for each image
        # for i in range(labels.shape[0]):
        #     # get as numpy arrays
        #     label = labels[i]
        #     pred = bb8dets[i]
        #     loc_mae_pixel_per_image = []
        #     bb8_mae_pixel_per_image = []
        #     # calculate for each class
        #     while (pred.shape[0] > 0):
        #         cid = int(pred[0, 0])
        #         indices = np.where(pred[:, 0].astype(int) == cid)[0]
        #         if cid < 0:
        #             pred = np.delete(pred, indices, axis=0)
        #             continue
        #         dets = pred[indices]
        #         pred = np.delete(pred, indices, axis=0)
        #
        #         # ground-truths
        #         label_indices = np.where(label[:, 0].astype(int) == cid)[0]
        #         gts = label[label_indices, :]
        #         label = np.delete(label, label_indices, axis=0)
        #         if gts.size > 0:
        #             found = [False] * gts.shape[0]
        #             for j in range(dets.shape[0]):
        #                 # compute overlaps
        #                 ious = iou(dets[j, 2:6], gts[:, 1:5])
        #                 ovargmax = np.argmax(ious)
        #                 ovmax = ious[ovargmax]
        #                 if ovmax > self.ovp_thresh:
        #                     if not found[ovargmax]:
        #                         loc_mae_pixel_per_image.append(np.abs((dets[j, 2:6] - gts[ovargmax, 1:5]) * 300))   # tp
        #                         bb8_mae_pixel_per_image.append(np.abs((dets[j, 6:22] - gts[ovargmax, 8:24]) * 300))
        #                         found[ovargmax] = True
        #                     else:
        #                         # duplicate
        #                         pass  # fp
        #
        #     loc_mae_pixel.append(np.mean(loc_mae_pixel_per_image, axis=1))
        #     bb8_mae_pixel.append(np.mean(bb8_mae_pixel_per_image, axis=1))


        valid_count = np.sum(cls_label >= 0)
        box_count = np.sum(cls_label > 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        # in case you have a 'other' class
        label[np.where(label >= cls_prob.shape[1])] = 0
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1])).asnumpy()
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # loc_smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += box_count * 4
        # loc_mae
        self.sum_metric[2] += np.sum(loc_mae)
        self.num_inst[2] += box_count * 4
        # loc_mae_pixel
        self.sum_metric[3] += np.sum(loc_mae_pixel)
        self.num_inst[3] += loc_mae_pixel.size
        # bb8_smoothl1loss
        self.sum_metric[4] += np.sum(bb8_loss)
        self.num_inst[4] += box_count * 16
        # bb8_mae
        self.sum_metric[5] += np.sum(bb8_mae)
        self.num_inst[5] += box_count * 16
        # bb8_mae_pixel
        self.sum_metric[6] += np.sum(bb8_mae_pixel)
        self.num_inst[6] += bb8_mae_pixel.size


    def validate6Dpose(self, gt_pose, instance_bb8det=None, model_id=None):
        gt_pose = np.reshape(gt_pose, newshape=(4,4))
        model_objx_info = self.models_info[instance_bb8det[0] + model_id]

        min_x = model_objx_info['min_x'] * self.scale_to_meters
        min_y = model_objx_info['min_y'] * self.scale_to_meters
        min_z = model_objx_info['min_z'] * self.scale_to_meters
        size_x = model_objx_info['size_x'] * self.scale_to_meters
        size_y = model_objx_info['size_y'] * self.scale_to_meters
        size_z = model_objx_info['size_z'] * self.scale_to_meters
        max_x = min_x + size_x
        max_y = min_y + size_y
        max_z = min_z + size_z

        BoundingBox = np.zeros(shape=(8, 3))
        BoundingBox[0, :] = np.array([min_x, min_y, min_z])
        BoundingBox[1, :] = np.array([min_x, min_y, max_z])
        BoundingBox[2, :] = np.array([min_x, max_y, max_z])
        BoundingBox[3, :] = np.array([min_x, max_y, min_z])

        BoundingBox[4, :] = np.array([max_x, min_y, min_z])
        BoundingBox[5, :] = np.array([max_x, min_y, max_z])
        BoundingBox[6, :] = np.array([max_x, max_y, max_z])
        BoundingBox[7, :] = np.array([max_x, max_y, min_z])
        Xworld = np.reshape(BoundingBox, newshape=(-1, 3, 1))

        Ximg_gt_pix = np.dot(gt_pose[0:3, 0:3], BoundingBox.T) + gt_pose[0:3, 3:4]
        Ximg_gt_pix /= Ximg_gt_pix[2, :]
        Ximg_gt_pix = np.dot(self.cam_intrinsic[:, 0:3], Ximg_gt_pix)
        Ximg_gt_pix = Ximg_gt_pix[0:2, :].T
        Ximg_gt_pix = np.reshape(Ximg_gt_pix, newshape=(-1, 2, 1))

        Ximg_pix = instance_bb8det[6:22]
        Ximg_pix = np.reshape(Ximg_pix, newshape=(-1, 2, 1))
        img_shape = np.array([[[640.], [480.]]])  # 1x2x1
        Ximg_pix = Ximg_pix * img_shape

        Ximg_noised_pix = Ximg_gt_pix + np.random.normal(loc=0.0, scale=1.0, size=(8, 2, 1))
        epnpSolver = EPnP()
        error, Rt, Cc, Xc = epnpSolver.efficient_pnp_gauss(Xworld, Ximg_pix, self.cam_intrinsic)
        out = {"R": Rt[0:3, 0:3],
               "t": Rt[0:3, 3:4]}

        # absolute pose error
        rot_error = pose_error.re(R_est=out["R"], R_gt=gt_pose[0:3, 0:3]) / np.pi * 180.
        trans_error = pose_error.te(t_est=out["t"], t_gt=gt_pose[0:3, 3:4]) / 0.01

        return out


    def calculate6Dpose(self, instance_bb8det=None, model_id=None):
        model_objx_info = self.models_info[instance_bb8det[0] + model_id]

        min_x = model_objx_info['min_x'] * self.scale_to_meters
        min_y = model_objx_info['min_y'] * self.scale_to_meters
        min_z = model_objx_info['min_z'] * self.scale_to_meters
        size_x = model_objx_info['size_x'] * self.scale_to_meters
        size_y = model_objx_info['size_y'] * self.scale_to_meters
        size_z = model_objx_info['size_z'] * self.scale_to_meters
        max_x = min_x + size_x
        max_y = min_y + size_y
        max_z = min_z + size_z

        BoundingBox = np.zeros(shape=(8, 3))
        BoundingBox[0, :] = np.array([min_x, min_y, min_z])
        BoundingBox[1, :] = np.array([min_x, min_y, max_z])
        BoundingBox[2, :] = np.array([min_x, max_y, max_z])
        BoundingBox[3, :] = np.array([min_x, max_y, min_z])

        BoundingBox[4, :] = np.array([max_x, min_y, min_z])
        BoundingBox[5, :] = np.array([max_x, min_y, max_z])
        BoundingBox[6, :] = np.array([max_x, max_y, max_z])
        BoundingBox[7, :] = np.array([max_x, max_y, min_z])
        Xworld = np.reshape(BoundingBox, newshape=(-1, 3, 1))

        Ximg_pix = instance_bb8det[6:22]
        Ximg_pix = np.reshape(Ximg_pix, newshape=(-1, 2, 1))
        img_shape = np.array([[[640.], [480.]]])  # 1x2x1
        Ximg_pix = Ximg_pix * img_shape

        epnpSolver = EPnP()
        error, Rt, Cc, Xc = epnpSolver.efficient_pnp_gauss(Xworld, Ximg_pix, self.cam_intrinsic)
        out = {"R": Rt[0:3, 0:3],
        "t": Rt[0:3, 3:4]}

        return out

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
