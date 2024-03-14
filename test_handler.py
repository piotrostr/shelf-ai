import base64
import pickle

from handler import YOLOHandler  # Adjust the import path according to your setup


def test_handler():
    """
    test_handler requires a sample_image.jpg in the current directory

    it writes the indermediate output in case of save=True

    the test does not test for CUDA availability
    """
    # set to true in testing, leaving false for prod
    save = False
    plot = False
    # Initialize the handler
    handler = YOLOHandler()
    handler.initialize(
        context=None
    )  # Passing `None` as context, assuming it's not used in your initialize method

    # Open and prepare the image
    with open("./sample_image.jpg", "rb") as f:
        image_data = f.read()
    # Mimic an inference request payload
    data = [{"body": [{"data": base64.b64encode(image_data).decode("utf-8")}]}]

    # Preprocess
    preprocessed_data = handler.preprocess(data)

    # Inference
    inference_output = handler.inference(preprocessed_data)

    if save:
        with open("inference-output.pkl", "wb") as f:
            pickle.dump(inference_output, f)

    # Postprocess
    postprocessed_output = handler.postprocess(inference_output)

    assert postprocessed_output

    if plot:
        import cv2

        plotted = inference_output[0].plot()
        cv2.imshow("image", plotted)
        cv2.waitKey(0)

    if save:
        with open("output.pkl", "wb") as f:
            pickle.dump(postprocessed_output, f)

    assert len(inference_output[0].boxes) > 5

    assert len(postprocessed_output[0]) > 5

    assert len(postprocessed_output[0]) == len(inference_output[0].boxes)
