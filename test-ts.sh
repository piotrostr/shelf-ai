#!/bin/bash

curl -X POST http://localhost:8080/predictions/retail-yolo -T ./sample_image.jpg

