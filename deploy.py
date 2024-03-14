"""
deploy.py requires an image of the built model, can be obtained by running cloud build

user has to ensure all of the configuration params are matching

model upload + endpoint + model deploy can take around ~15 mins
""" 
from google.cloud import aiplatform


PROJECT_ID = "vertex-ai-playground-402513"
REGION = "europe-west4"

IMAGE_URI = "europe-central2-docker.pkg.dev/vertex-ai-playground-402513/shelf-ai/retail-yolo:0.2.0"
MODEL_DISPLAY_NAME = "retail-yolo-v0.2.0"
MODEL_DESCRIPTION = "Object detection model for retail products"
MODEL_NAME = "retail-yolo"
HEALTH_ROUTE = "/ping"
PREDICT_ROUTE = "/predictions/retail-yolo"
SERVING_CONTAINER_PORTS = [8080]

ENDPOINT_DISPLAY_NAME = "retail-yolo-endpoint"

TRAFFIC_PERCENTAGE = 100
MACHINE_TYPE = "n1-standard-4"
MIN_NODES = 1
MAX_NODES = 3
SYNC = True
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

if __name__ == "__main__":
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        description=MODEL_DESCRIPTION,
        serving_container_image_uri=IMAGE_URI,
        serving_container_predict_route=PREDICT_ROUTE,
        serving_container_health_route=HEALTH_ROUTE,
        serving_container_ports=SERVING_CONTAINER_PORTS,
    )

    # To use this Model in another session:
    # model = aiplatform.Model('projects/352528412502/locations/europe-west4/models/3770520040760147968@1')

    model.wait()

    print(model)

    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
    )

    res = model.deploy(
        endpoint=endpoint,
        machine_type=MACHINE_TYPE,
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        traffic_percentage=TRAFFIC_PERCENTAGE,
        sync=SYNC,
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
    )

    model.wait()

    print(res)
