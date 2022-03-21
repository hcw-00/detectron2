import os
import pickle

from typing import List, Dict, Tuple
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.ops._box_convert import _box_xyxy_to_cxcywh
from detectron2.modeling.meta_arch.dense_detector import permute_to_N_HWA_K

# TODO add anchor generator

def anchor_generator_forward_for_onnx(self, features):
    grid_sizes = [feature_map.shape[-2:] for feature_map in features]
    anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
    return anchors_over_all_feature_maps

def xyxy_to_xywh(inputs):
    inputs = inputs[0]
    inputs = _box_xyxy_to_cxcywh(inputs)
    inputs = torch.unsqueeze(inputs, 0)
    return inputs

def preprocess_image_onnx(self, batched_inputs: List[Dict[str, Tensor]]):
    """
    Normalize, pad and batch the input images.
    """
    # images = [x["image"].to(self.device) for x in batched_inputs]
    # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
    # images = ImageList.from_tensors(images, self.backbone.size_divisibility)
    pixel_mean = [103.530, 116.280, 123.675]
    pixel_std = [57.375, 57.12, 58.395]
    images = transforms.Normalize(mean=pixel_mean, std=pixel_std)(batched_inputs)    
    return images

def forward_onnx(self, batched_inputs: List[Dict[str, Tensor]]):
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images)
    features = [features[f] for f in self.head_in_features]
    predictions = self.head(features)
    results = self.forward_inference(images, features, predictions)
    return results


def forward_inference_onnx(
        self, images, features: List[Tensor], predictions: List[List[Tensor]]
    ):
    pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
        predictions, [self.num_classes, 4]
    )
    anchors_ = self.anchor_generator(features)
    anchor_file_name = f"anchor_{images.shape[-1]}.pkl"
    if not os.path.exists(anchor_file_name):
        with open(anchor_file_name, 'wb') as f:
            pickle.dump(anchors_, f)
    with open(anchor_file_name, 'rb') as f:
        anchors = pickle.load(f)
    scores_per_image = pred_logits
    deltas_per_image = pred_anchor_deltas
    anchors, box_cls, box_delta = self.inference_single_image(
        anchors, scores_per_image, deltas_per_image
    )
    pred_logits = torch.cat(pred_logits, 1)
    anchors = xyxy_to_xywh(anchors)
    return anchors, box_delta, pred_logits, box_cls

def inference_single_image_onnx(
        self,
        anchors: List, #[Boxes],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        #image_size: Tuple[int, int],
    ):
    anchors = torch.unsqueeze(torch.cat(anchors), 0)
    box_cls = torch.cat(box_cls, 1)
    box_delta = torch.cat(box_delta, 1)
    return anchors, box_cls, box_delta
