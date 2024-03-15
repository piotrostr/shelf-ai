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
    save = True
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
    request = [
        {
            "body": {
                "instances": [
                    {"data": base64.b64encode(image_data).decode("utf-8")},
                    {"data": base64.b64encode(image_data).decode("utf-8")},
                    {"data": base64.b64encode(image_data).decode("utf-8")},
                    {"data": base64.b64encode(image_data).decode("utf-8")},
                    {"data": base64.b64encode(image_data).decode("utf-8")},
                ],
            }
        },
    ]

    # Preprocess
    preprocessed_data = handler.preprocess(request)

    # Inference
    inference_outputs = handler.inference(preprocessed_data)

    assert len(inference_outputs[0].boxes) > 5

    if save:
        with open("inference-output.pkl", "wb") as f:
            pickle.dump(inference_outputs, f)

    # Postprocess
    postprocessed_outputs = handler.postprocess(inference_outputs)

    assert postprocessed_outputs

    if plot:
        import cv2

        plotted = inference_outputs[0].plot()
        cv2.imshow("image", plotted)
        cv2.waitKey(0)

    if save:
        with open("output.pkl", "wb") as f:
            pickle.dump(postprocessed_outputs, f)

    # got to unnset, this is a single request
    # it returns a nested array since we don't return 'instances'
    # but a list of requests that contain separates 'instances'
    # we will be comparing against the inference outputs,
    # which don't support nested requests due to Ultralytics output format
    postprocessed_outputs = postprocessed_outputs[0]

    assert "predictions" in postprocessed_outputs

    # verify postprocessing
    for instance_output, inference_output in zip(
        postprocessed_outputs["predictions"], inference_outputs
    ):
        assert len(instance_output) == len(inference_output.boxes)
