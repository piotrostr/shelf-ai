FROM nvcr.io/nvidia/tensorrt:24.03-py3

RUN apt update && apt install -y libgl1-mesa-glx

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY server.py .
COPY model.py .
COPY run.sh .
COPY ./retail-yolo.engine .

EXPOSE 3000/tcp

CMD ["./run.sh"]
