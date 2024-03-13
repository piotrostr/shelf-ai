FROM pytorch/torchserve:0.9.0-gpu

# install deps
COPY ./requirements.txt /home/model-server/

RUN pip install --no-cache-dir -r /home/model-server/requirements.txt

COPY ./handler.py /home/model-server/
COPY ./retail-yolo.pt /home/model-server/

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=retail-yolo \
  --version=1.0 \
  --serialized-file=/home/model-server/retail-yolo.pt \
  --handler=/home/model-server/handler.py \
  --export-path=/home/model-server/model-store

# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "retail-yolo=retail-yolo.mar", \
     "--model-store", \
     "/home/model-server/model-store"]
