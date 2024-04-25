from ultralytics import YOLO

if __name__ == "__main__":
    yolo = YOLO("./retail-yolo.engine")
    yolo.predict("./sample_image.jpg")
