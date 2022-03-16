#! /bin/bash

python /home/appuser/detectron2/tools/deploy/export_model.py \
--format onnx --export-method onnx \
--config-file /home/appuser/detectron2_repo/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml