#!/bin/bash

docker build -t shelf-ai-ws .

docker run -it -p 3000:3000 --rm --gpus all shelf-ai-ws
