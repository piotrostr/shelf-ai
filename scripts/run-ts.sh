#!/bin/bash

torchserve \
  --start \
  --ncs \
  --model-store model_store \
  --models retail-yolo=retail-yolo.mar
