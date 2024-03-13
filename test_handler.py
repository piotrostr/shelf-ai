import pickle
import cv2
from handler import YOLOHandler  # Adjust the import path according to your setup


def test_handler():
    save = True
    plot = True
    # Initialize the handler
    handler = YOLOHandler()
    handler.initialize(
        context=None
    )  # Passing `None` as context, assuming it's not used in your initialize method

    # Open and prepare the image
    with open("./zidane.jpg", "rb") as f:
        image_data = f.read()
    # Mimic an inference request payload
    data = [{"data": image_data}]

    # Preprocess
    preprocessed_data = handler.preprocess(data)

    # Inference
    inference_output = handler.inference(preprocessed_data)

    with open("inference-output.pkl", "wb") as f:
        pickle.dump(inference_output, f)

    # Postprocess
    postprocessed_output = handler.postprocess(inference_output)

    assert postprocessed_output

    if plot:
        plotted = inference_output[0].plot()
        cv2.imshow("image", plotted)
        cv2.waitKey(0)

    if save:
        with open("output.pkl", "wb") as f:
            pickle.dump(postprocessed_output, f)

    assert False
