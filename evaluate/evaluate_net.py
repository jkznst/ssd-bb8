import os
import sys
import importlib
import mxnet as mx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging
from tqdm import tqdm

from dataset.iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric, PoseMetric
from symbol.symbol_factory import get_symbol
from MultiBoxDetection import BB8MultiBoxDetection



def show_BB8(image, gt_BB8_image_coordinates, pred_BB8_image_coordinates, plot_path):
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    # fig = plt.figure()
    # (3L, 256L, 256L) => (256L, 256L, 3L)
    img = image
    img = cv2.resize(img, dsize=(640, 480))
    # img = np.flip(img, axis=2)
    plt.imshow(img)
    gt_BB8_image_coordinates *= np.array([640, 480]).reshape((2,1))
    pred_BB8_image_coordinates *= np.array([640, 480]).reshape((2,1))

    rect0 = plt.Line2D([gt_BB8_image_coordinates[0, 0], gt_BB8_image_coordinates[0, 1],
                        gt_BB8_image_coordinates[0, 2], gt_BB8_image_coordinates[0, 3],
                        gt_BB8_image_coordinates[0, 0]],
                       [gt_BB8_image_coordinates[1, 0], gt_BB8_image_coordinates[1, 1],
                        gt_BB8_image_coordinates[1, 2], gt_BB8_image_coordinates[1, 3],
                        gt_BB8_image_coordinates[1, 0]],
                       linewidth=2, color='green')
    rect1 = plt.Line2D([gt_BB8_image_coordinates[0, 4], gt_BB8_image_coordinates[0, 5],
                        gt_BB8_image_coordinates[0, 6], gt_BB8_image_coordinates[0, 7],
                        gt_BB8_image_coordinates[0, 4]],
                       [gt_BB8_image_coordinates[1, 4], gt_BB8_image_coordinates[1, 5],
                        gt_BB8_image_coordinates[1, 6], gt_BB8_image_coordinates[1, 7],
                        gt_BB8_image_coordinates[1, 4]],
                       linewidth=2, color='green')
    rect2 = plt.Line2D([gt_BB8_image_coordinates[0, 0], gt_BB8_image_coordinates[0, 4]],
                       [gt_BB8_image_coordinates[1, 0], gt_BB8_image_coordinates[1, 4]],
                       linewidth=2, color='green')
    rect3 = plt.Line2D([gt_BB8_image_coordinates[0, 1], gt_BB8_image_coordinates[0, 5]],
                       [gt_BB8_image_coordinates[1, 1], gt_BB8_image_coordinates[1, 5]],
                       linewidth=2, color='green')
    rect4 = plt.Line2D([gt_BB8_image_coordinates[0, 2], gt_BB8_image_coordinates[0, 6]],
                       [gt_BB8_image_coordinates[1, 2], gt_BB8_image_coordinates[1, 6]],
                       linewidth=2, color='green')
    rect5 = plt.Line2D([gt_BB8_image_coordinates[0, 3], gt_BB8_image_coordinates[0, 7]],
                       [gt_BB8_image_coordinates[1, 3], gt_BB8_image_coordinates[1, 7]],
                       linewidth=2, color='green')
    axes.add_line(rect0)
    axes.add_line(rect1)
    axes.add_line(rect2)
    axes.add_line(rect3)
    axes.add_line(rect4)
    axes.add_line(rect5)

    rect6 = plt.Line2D([pred_BB8_image_coordinates[0, 0], pred_BB8_image_coordinates[0, 1],
                        pred_BB8_image_coordinates[0, 2], pred_BB8_image_coordinates[0, 3],
                        pred_BB8_image_coordinates[0, 0]],
                       [pred_BB8_image_coordinates[1, 0], pred_BB8_image_coordinates[1, 1],
                        pred_BB8_image_coordinates[1, 2], pred_BB8_image_coordinates[1, 3],
                        pred_BB8_image_coordinates[1, 0]],
                       linewidth=2, color='blue')
    rect7 = plt.Line2D([pred_BB8_image_coordinates[0, 4], pred_BB8_image_coordinates[0, 5],
                        pred_BB8_image_coordinates[0, 6], pred_BB8_image_coordinates[0, 7],
                        pred_BB8_image_coordinates[0, 4]],
                       [pred_BB8_image_coordinates[1, 4], pred_BB8_image_coordinates[1, 5],
                        pred_BB8_image_coordinates[1, 6], pred_BB8_image_coordinates[1, 7],
                        pred_BB8_image_coordinates[1, 4]],
                       linewidth=2, color='blue')
    rect8 = plt.Line2D([pred_BB8_image_coordinates[0, 0], pred_BB8_image_coordinates[0, 4]],
                       [pred_BB8_image_coordinates[1, 0], pred_BB8_image_coordinates[1, 4]],
                       linewidth=2, color='blue')
    rect9 = plt.Line2D([pred_BB8_image_coordinates[0, 1], pred_BB8_image_coordinates[0, 5]],
                       [pred_BB8_image_coordinates[1, 1], pred_BB8_image_coordinates[1, 5]],
                       linewidth=2, color='blue')
    rect10 = plt.Line2D([pred_BB8_image_coordinates[0, 2], pred_BB8_image_coordinates[0, 6]],
                       [pred_BB8_image_coordinates[1, 2], pred_BB8_image_coordinates[1, 6]],
                       linewidth=2, color='blue')
    rect11 = plt.Line2D([pred_BB8_image_coordinates[0, 3], pred_BB8_image_coordinates[0, 7]],
                       [pred_BB8_image_coordinates[1, 3], pred_BB8_image_coordinates[1, 7]],
                       linewidth=2, color='blue')
    axes.add_line(rect6)
    axes.add_line(rect7)
    axes.add_line(rect8)
    axes.add_line(rect9)
    axes.add_line(rect10)
    axes.add_line(rect11)
    axes.axes.get_xaxis().set_visible(False)
    axes.axes.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig(plot_path)
    plt.close(fig)


def evaluate_net(net, path_imgrec, num_classes, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 path_imglist="", nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False, class_names=None,
                 voc07_metric=False, frequent=20):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    frequent : int
        frequency to print out validation status
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    #model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = DetRecordIter(path_imgrec, batch_size, data_shape, mean_pixels=mean_pixels,
                              label_pad_width=350, path_imglist=path_imglist, **cfg.valid)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if net is None:
        net = load_net
    else:
        net = get_symbol(net, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    # run evaluation
    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names,
                                roc_output_path=os.path.join(os.path.dirname(model_prefix), 'roc'))
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names,
                            roc_output_path=os.path.join(os.path.dirname(model_prefix), 'roc'))

    metric = PoseMetric(LINEMOD_path='/data/ZHANGXIN/DATASETS/SIXD_CHALLENGE/LINEMOD/', classes=class_names)

    # for i in range(1, 16):
    #     if class_names[0] == 'obj_{:02d}'.format(i):
    #         model_id = i
    #         break
    # # visualize bb8 results
    # for nbatch, eval_batch in tqdm(enumerate(eval_iter)):
    #     mod.forward(eval_batch)
    #     preds = mod.get_outputs(merge_multi_context=True)
    #
    #     labels = eval_batch.label[0].asnumpy()
    #     # get generated multi label from network
    #     cls_prob = preds[0]
    #     loc_pred = preds[4]
    #     bb8_pred = preds[5]
    #     anchors = preds[6]
    #
    #     bb8dets = BB8MultiBoxDetection(cls_prob, loc_pred, bb8_pred, anchors, nms_threshold=0.5, force_suppress=False,
    #                                   variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
    #     bb8dets = bb8dets.asnumpy()
    #
    #     for idx in range(bb8dets.shape[0]):
    #         if bb8dets[idx, 0, 0] == -1:
    #             continue
    #         else:
    #             # for LINEMOD dataset, for each image only select the first det
    #             image = eval_batch.data[0][idx].asnumpy()
    #             image += np.array(mean_pixels).reshape((3,1,1))
    #             image = np.transpose(image, axes=(1,2,0))
    #             gtbb8 = eval_batch.label[0][idx][0].asnumpy()
    #
    #             show_BB8(image / 255., gtbb8[8:24].reshape((8,2)).T, bb8dets[idx, 0, 6:].reshape((8,2)).T,
    #                      plot_path='./output/bb8results/{:02d}_{:04d}'.format(model_id, nbatch * batch_size + idx))

    # quantitive results
    results = mod.score(eval_iter, metric, num_batch=None,
                        batch_end_callback=mx.callback.Speedometer(batch_size,
                                                                   frequent=frequent,
                                                                   auto_reset=False))

    results_save_path = os.path.join(os.path.dirname(model_prefix), 'evaluate_results')
    with open(results_save_path, 'w') as f:
        for k, v in results:
            print("{}: {}".format(k, v))
            f.write("{}: {}\n".format(k, v))
        f.close()

    # net_output = mod.predict(eval_iter)
    # print(1)
