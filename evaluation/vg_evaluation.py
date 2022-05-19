from email.policy import default
import os, io
import numpy as np

import copy
import torch
import logging
import pickle as cPickle
import itertools
import contextlib
from pycocotools.coco import COCO
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from .vg_eval import vg_eval

class VGEvaluator(DatasetEvaluator):
    """
        Evaluate object proposal, instance detection
        outputs using VG's metrics and APIs.
    """
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._output_dir = output_dir

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_vg_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._classes = ['__background__']
        self._class_to_ind = {}
        self._class_to_ind[self._classes[0]] = 0
        with open(os.path.join('evaluation/objects_vocab.txt')) as f:
            count = 1
            for object in f.readlines():
                names = [n.lower().strip() for n in object.split(',')]
                self._classes.append(names[0])
                for n in names:
                    self._class_to_ind[n] = count
                    ## TODO drigoni: tmp
                    #if n not in self._class_to_ind.keys():
                    #    self._class_to_ind[n] = count
                    #else:
                    #    print("Error: class name already present:", n)
                    #    exit(1)
                count += 1

        # Load attributes
        self._attributes = ['__no_attribute__']
        self._attribute_to_ind = {}
        self._attribute_to_ind[self._attributes[0]] = 0
        with open(os.path.join('evaluation/attributes_vocab.txt')) as f:
            count = 1
            for att in f.readlines():
                names = [n.lower().strip() for n in att.split(',')]
                self._attributes.append(names[0])
                for n in names:
                    self._attribute_to_ind[n] = count
                count += 1

        self.cat_map, self.cat_new_labels, self.cat_old_labels = self._get_categories_mapping()
        self.roidb, self.image_index = self.gt_roidb(self._coco_api)
        assert len(self._classes)-1 == len(self.cat_new_labels) # self._classes has __background__ class


    def _get_categories_mapping(self, labels_file='evaluation/objects_vocab.txt'):
        '''
        This function creates the mapping function from the old classes to the new ones.
        :param labels_file: new classes.
        :return: mapping function, index to labels name for new classes, index to labels name for old classes
        '''
        # loading cleaned classes
        print("Loading cleaned Visual Genome classes: {} .".format(labels_file))
        with open(labels_file, 'r') as f:
            cleaned_labels = f.readlines() # NOTE: removed __background__ class from file
        # remove new line symbol and leading/trailing spaces.
        cleaned_labels = [i.strip('\n').strip() for i in cleaned_labels]
        # make dictionary
        cleaned_labels = {id+1: label for id, label in enumerate(cleaned_labels)}     # [1, 1600]
        # get previously labels from the same file and make the mapping function
        map_fn = dict()
        old_labels = dict()
        for new_label_id, new_label_str in cleaned_labels.items():
            new_label_id = int(new_label_id)
            for piece in new_label_str.split(','):
                tmp = piece.split(':')
                assert len(tmp) == 2
                old_label_id = int(tmp[0])
                old_label_str = tmp[1]
                # we need to avoid overriding of same ids like: 17:stop sign,17:stopsign
                if old_label_id not in old_labels.keys():
                    old_labels[old_label_id] = old_label_str
                    map_fn[old_label_id] = new_label_id
                else:
                    print('Warning: label already present for {}:{}. Class {} ignored. '.format(old_label_id,
                                                                                                old_labels[old_label_id],
                                                                                                old_label_str))
        assert len(old_labels) == 1600
        assert len(old_labels) == len(map_fn)
        # print(old_labels[1590], map_fn[1590], cleaned_labels[map_fn[1590]])
        return map_fn, cleaned_labels, old_labels     # all in [1, 1600]

    def _tasks_from_config(self, cfg):
        """attribute_ids
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def gt_roidb(self, dataset):
        roidb = []
        image_index = dataset.imgToAnns.keys()
        for img_index in dataset.imgToAnns:
            tmp_dict = {}
            num_objs = len(dataset.imgToAnns[img_index])
            bboxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_attributes = np.zeros((num_objs, 16), dtype=np.int32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            for ind, item in enumerate(dataset.imgToAnns[img_index]):
                bboxes[ind, :] = item['bbox']
                old_Category = item['category_id'] # NOTE: [0: num_old_Classes-1]
                # gt_classes[ind] = item['category_id'] + 1 # NOTE
                gt_classes[ind] = self.cat_map[old_Category + 1]  # -1 not necessary because we need to add 1 according to original code
                # drigoni: check
                # print("GT classes: {}:{} || {}:{}".format(old_Category, self.cat_old_labels[old_Category + 1],
                                                            # gt_classes[ind], self.cat_new_labels[gt_classes[ind] + 1] )) # TODO drigoni tmp
                attrs = item.get("attribute_ids", None)
                if attrs:
                    for j, attr in enumerate(item['attribute_ids']):
                        gt_attributes[ind, j] = attr
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            tmp_dict['boxes'] = bboxes
            tmp_dict['gt_attributes'] = gt_attributes
            tmp_dict['gt_classes'] = gt_classes
            roidb.append(tmp_dict)
        return roidb, image_index

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["boxes"] = instances.pred_boxes.tensor.numpy()
                old_pred_classes = instances.pred_classes.numpy()
                prediction["labels"] = np.array([self.cat_map[i + 1]-1 for i in old_pred_classes])

                # drigoni: check
                # for old, new in zip(old_pred_classes, prediction["labels"]):
                    # print("Pred classes: {}:{} || {}:{}".format(old, self.cat_old_labels[old + 1],
                                                                # new, self.cat_new_labels[new + 1] )) # TODO drigoni tmp

                old_scores = instances.scores.numpy()
                prediction["scores"] = old_scores
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        # self._predictions = torch.load(os.path.join(self._output_dir, "instances_predictions.pth"))

        if len(self._predictions) == 0:
            self._logger.warning("[VGEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        self._eval_vg()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_vg(self):
        self.write_voc_results_file(self._predictions, output_dir=self._output_dir)
        self.do_python_eval(self._output_dir)

    def write_voc_results_file(self, predictions, output_dir):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing "{}" vg result file'.format(cls))
            filename = self.get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for pred_ind, item in enumerate(predictions):
                    scores = item["scores"]
                    labels = item["labels"] + 1     # due to __background__ class
                    # TODO drigoni: check
                    # for tmp_i in labels:
                    #     if tmp_i > 878:
                    #         print('Error: ', labels)
                    #         exit(1)
                    bbox = item["boxes"]
                    if cls_ind not in labels:
                        continue
                    dets = bbox[labels==cls_ind]
                    scores = scores[labels==cls_ind]
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(item["image_id"]), scores[k],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def get_vg_results_file_template(self, output_dir, pickle=True, eval_attributes = False):
        filename = 'detections_vg'+'_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def do_python_eval(self, output_dir, pickle=True, eval_attributes = False, by_npos = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        if eval_attributes:
            classes = self._attributes
        else:
            classes = self._classes
        print("Number of classes: {}. ".format(len(classes)))
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self.get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, self.roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1 and not (type(prec) == int and type(rec) == int and prec+rec ==0):
                f = np.nan_to_num((prec * rec) / (prec + rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                                  'scores': scores, 'npos': npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh != 0])
        thresh[thresh == 0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_vg.txt'
        else:
            filename = 'object_thresholds_vg.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        print("Number of nposs: {}.".format(len(nposs)))
        print("Sum of nposs: {}.".format(sum(nposs)))
        if True:
            import json
            import matplotlib.pyplot as plt
            old_aps = copy.deepcopy(aps)
            old_nposs = copy.deepcopy(nposs)
            # saving all scores
            filename = 'all_AP_scores_by_GTs.txt'
            path = os.path.join(output_dir, filename)
            results = defaultdict(list)
            for npos, ap in zip(old_nposs, old_aps):
                results[npos].append(ap)
            print("Number of results: {}.".format(len(results)))
            results = {key: sum(val_list)/max(len(val_list), 1) for key, val_list in results.items()}
            print("Number of results after division: {}.".format(len(results)))
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
                print('Saved file: {}'.format(path))
            # count cumulative AP
            scores = []
            wscores = []
            points = [10, 30, 60, 100, 200, 300, 400, 600, 800, 1000, 2000, 3000]
            for i in points:
                print('-- Classes with at max {} gts. '.format(i))
                aps = []
                nposs = []
                for ap, npos in zip(old_aps, old_nposs):
                    if npos <= i:
                        aps.append(ap)
                        nposs.append(npos)
                weights = np.array(nposs)
                weights /= weights.sum()
                print('Mean AP = {:.4f}'.format(np.mean(aps)))
                print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
                print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
                scores.append(np.mean(aps))
                wscores.append(np.average(aps, weights=weights))
            # save a dump of the results
            filename = 'cumulative_AP_scores_by_GTs.txt'
            path = os.path.join(output_dir, filename)
            results = {p: (s, ws) for p, s, ws in zip(points, scores, wscores)}
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
            print('Saved file: {}'.format(path))
        else:
            weights = np.array(nposs)
            weights /= weights.sum()
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
            print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap, npos in zip(aps, nposs):
        #     print('{:.3f}\t{:.3f}'.format(ap, npos))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')
