from google.cloud import aiplatform

if __name__ == "__main__":
    ENDPOINT_ID="2553457425835360256"
    PROJECT_ID="352528412502"

    aiplatform.init(
        project=PROJECT_ID,
        location="europe-west4"
    )


    endpoint = aiplatform.Endpoint(
        f"projects/{PROJECT_ID}/locations/europe-west4/endpoints/{ENDPOINT_ID}"
    )

    image_path = "./sample_image.jpg"
    import base64

    with open(image_path, "rb") as f:
        data = {"data": base64.b64encode(f.read()).decode()}

    print(endpoint.predict(instances=[data]))
