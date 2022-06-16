# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import Instances, BoxMode, Boxes
from detectron2.modeling import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference, fast_rcnn_inference_single_image


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], GT_proposals=True):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            # NOTE drigoni: function that uses GT boxes as proposals
            images = model.preprocess_image(inputs)     # get the images data
            features = model.backbone(images.tensor)    # get the whole images features
            if GT_proposals:
                # 1) generate the proposal boxes with the GT classes
                all_images_size = [(i['height'], i['width']) for i in inputs]
                # get all annotations in batch
                all_annotations = [i['annotations'] for i in inputs]
                # get all boxes for annotations in the batch
                all_boxes = [[BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos] 
                            for annos in all_annotations]
                all_boxes = [Boxes(boxes) for boxes in all_boxes]
                proposals = []
                for img_size, img_boxes in zip(all_images_size, all_boxes):
                    instance = Instances(img_size)
                    instance.proposal_boxes = img_boxes.to(model.device)
                    instance.objectness_logits = torch.tensor([10]*len(img_boxes)).to(model.device)
                    proposals.append(instance)
                
                # 2) extract features given the proposals.
                # pooled_features -> torch.Size([134, 2048]
                _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)
                # 3) extract per class logits and coordinates deltas
                # tupla (scores, proposal_deltas). scores.shape -> torch.Size([n_proposals, 1601])
                predictions = model.roi_heads.box_predictor(pooled_features)
                # 4) pred final boxes.
                # proposals coordinates without deltas are returned and we are using argmax. This implies n_proposals == final_boxes 
                pred_instances, _ = box_predictor_inference(model.roi_heads.box_predictor, predictions, proposals)
            else:
                # 1) generate the proposal with RPN
                proposals, losses  = model.proposal_generator(images, features, None)   # get all the proposals boxes from RPN
                # 2) extract features given the proposals.
                # pooled_features -> torch.Size([134, 2048]
                _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)
                # 3) extract per class logits and coordinates deltas
                # tupla (scores, proposal_deltas). scores.shape -> torch.Size([n_proposals, 1601])
                predictions = model.roi_heads.box_predictor(pooled_features)
                # 4) pred final boxes. Note: this function applies the sigmoid/softmax and the deltas on the proposals coordinates.
                # Each box with high probability is returned. This implies n_proposals <= final_boxes 
                pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)     # Apply sigmoid/softmax and applies deltas to proposals

            # 5) # Add new keys. Useless for this task.
            pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)  
            # 6) post-process results 
            outputs = model._postprocess(pred_instances, inputs, images.image_sizes)
            # NOTE drigoni: code for checking results
            # processed_results = []
            # for results_per_image, input_per_image, image_size, image_proposal in zip(pred_instances, inputs, images.image_sizes, proposals):
            #     # note that "r" and  "results_per_image" are the same
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])
            #     print("image_proposal", image_proposal)
            #     print("results_per_image", results_per_image)
            #     r = detector_postprocess(results_per_image, height, width)
            #     processed_results.append({"instances": r})
            # outputs = processed_results

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def box_predictor_inference(box_predictor, predictions, proposals):
    """
    Args:
        box_predictor: for example the fastrccn model
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were
            used to compute predictions. The ``proposal_boxes`` field is expected.
    Returns:
        list[Instances]: same as `fast_rcnn_inference`.
        list[Tensor]: same as `fast_rcnn_inference`.
    """
    boxes = box_predictor.predict_boxes(predictions, proposals)
    scores = box_predictor.predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    # return fast_rcnn_inference(
    #     boxes,
    #     scores,
    #     image_shapes,
    #     box_predictor.test_score_thresh,
    #     box_predictor.test_nms_thresh,
    #     box_predictor.test_topk_per_image,
    # )
    result_per_image = []
    for scores_per_image, boxes_per_image, image_shape, img_original_proposals in zip(scores, boxes, image_shapes, proposals):
        # original code
        # img_res = fast_rcnn_inference_single_image(boxes_per_image, scores_per_image, image_shape, 
        #                                             box_predictor.test_score_thresh, box_predictor.test_nms_thresh, box_predictor.test_topk_per_image)
        # Note that are n_classes+1 logits. 
        scores_per_image = scores_per_image[:, :-1]
        class_score, class_idx = torch.max(scores_per_image, dim=-1)
        img_res = Instances(image_shape)
        img_res.pred_boxes = img_original_proposals.proposal_boxes
        img_res.scores = class_score
        img_res.pred_classes = class_idx
        # indexes = 
        result_per_image.append((img_res, class_idx))
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]
