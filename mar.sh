#!/bin/bash

torch-model-archiver \
  --model-name retail-yolo \
  --version 1.0 \
  --serialized-file retail-yolo.pt \
  --handler handler.py

