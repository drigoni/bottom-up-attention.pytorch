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
import json

class VGEvaluator_classification(DatasetEvaluator):
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

        self.roidb, self.image_index = self.gt_roidb(self._coco_api)

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
                gt_classes[ind] = item['category_id'] + 1 # NOTE
                attrs = item.get("attribute_ids", None)
                if attrs:
                    for j, attr in enumerate(item['attribute_ids']):
                        gt_attributes[ind, j] = attr
            bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
            tmp_dict['boxes'] = bboxes
            # tmp_dict['gt_attributes'] = gt_attributes
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
                prediction["labels"] = instances.pred_classes.numpy()
                prediction["scores"] = instances.scores.numpy()
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
        self.do_python_eval(self._output_dir)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def do_python_eval(self, output_dir, pickle=True):
        # change interval of classes in the predicted classes. [1, ..]
        for pred_data in self._predictions:
            pred_data['labels'] = pred_data['labels'] + 1
        classes = self._classes

        # make useful dictionaries
        images = {pred['image_id']: pred for pred in self._predictions}
        assert len(images) == len(self._predictions)
        GTs = {idx: gt for idx, gt in zip(self.image_index, self.roidb)}
        assert len(GTs) == len(self.roidb)
        
        # group boxes by GT class
        data_by_class = {i+1: [] for i in range(len(classes) - 1)}
        all_hits = {i+1: [] for i in range(len(classes) - 1)}
        for image_idx in GTs.keys():
            pred_data = images[image_idx]
            GT_data = GTs[image_idx]
            for box_id in range(len(GT_data['boxes'])):
                tmp_dict = {}
                tmp_dict['GT_box'] = GT_data['boxes'][box_id]
                tmp_dict['GT_class'] = GT_data['gt_classes'][box_id]
                tmp_dict['pred_box'] = pred_data['boxes'][box_id]
                tmp_dict['pred_score'] = pred_data['scores'][box_id]
                tmp_dict['pred_class'] = pred_data['labels'][box_id]
                assert tmp_dict['GT_class'] in data_by_class.keys()
                data_by_class[tmp_dict['GT_class']].append(tmp_dict)
                all_hits[tmp_dict['GT_class']].append(1 if tmp_dict['GT_class'] == tmp_dict['pred_class'] else 0)
        # NOTE: 
        # 1) the number of GT boxes equals the number of pred boxes. 
        # 2) the GT boxes coordinates and the pred boxes coordinate are the same
        nposs = {k: float(len(v)) for k, v in all_hits.items()}
        accuracies = {k: np.mean(hits) if len(hits) > 0 else 0 for k, hits in all_hits.items()}

        # saving all scores
        filename = 'all_accuracy_by_category.json'
        path = os.path.join(output_dir, filename)
        results = defaultdict(list)
        for cls_id, accuracy in accuracies.items():
            values = [nposs[cls_id], accuracy]
            results[classes[cls_id]].append(values)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
            print('Saved file: {}'.format(path))

        weights = np.array(list(nposs.values()))
        weights /= weights.sum()
        print('Mean Accuracy = {:.4f}'.format(np.mean(list(accuracies.values()))))
        print('Weighted Mean AP = {:.4f}'.format(np.average(list(accuracies.values()), weights=weights)))
