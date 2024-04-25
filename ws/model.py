from ultralytics import YOLO


class Model:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, task="detect")

    def preprocess(self, data):
        return data

    def predict(self, image):
        return self.postprocess(
                self.model(
                    self.preprocess(image)
                )
            )

    def postprocess(self, data):
        return data

    def export_trt(self):
        res = self.model.export(format="engine")
        print(res)

