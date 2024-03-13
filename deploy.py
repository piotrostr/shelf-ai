from google.cloud import aiplatform

PROJECT_ID = "PROJECT_ID"
REGION = "europe-west4"

aiplatform.init(project=PROJECT_ID, location=REGION)

IMAGE_URI = "docker.io/piotrostr/shelf-ai-retail-yolo:latest"
MODEL_DISPLAY_NAME = "retail-yolo-v0.1.0"
MODEL_DESCRIPTION = "Object detection model for retail products"
MODEL_NAME = "retail-yolo"
HEALTH_ROUTE = "/ping"
PREDICT_ROUTE = "/predictions/retail-yolo"
SERVING_CONTAINER_PORTS = [8080]

model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    description=MODEL_DESCRIPTION,
    serving_container_image_uri=IMAGE_URI,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
    description=MODEL_DESCRIPTION,
)

model.wait()

print(model)

ENDPOINT_DISPLAY_NAME = "retail-yolo-endpoint"

endpoint = aiplatform.Endpoint.create(
    display_name=ENDPOINT_DISPLAY_NAME,
)

TRAFFIC_PERCENTAGE = 100
MACHINE_TYPE = "n1-standard-4"
MIN_NODES = 1
MAX_NODES = 3
SYNC = True
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

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
