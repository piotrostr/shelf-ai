#!/bin/bash

ENDPOINT_ID="2553457425835360256"
PROJECT_ID="352528412502"

cat > ./sample_request.json <<END
{
  "instances": [
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    }
  ]
}
END

res=$(curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west4/endpoints/${ENDPOINT_ID}:predict \
-d @./sample_request.json)

# write the json res
echo $res > ./res.json

cat > ./sample_request_batch.json <<END
{
  "instances": [
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    },
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    },
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    },
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    },
    {
      "data": "$(base64 -b 0 -i ./sample_image.jpg)"
    },
  ]
}
END

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/europe-west4/endpoints/${ENDPOINT_ID}:predict \
-d @./sample_request_batch.json
