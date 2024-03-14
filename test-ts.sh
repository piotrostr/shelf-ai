#!/bin/bash


curl -s -X POST http://localhost:8080/predictions/retail-yolo -T ./sample_image.jpg 

# check if works for JSON with base64 encoded image
cat > ./sample_request.json <<END
{
  "instances": [
    { 
      "data": "$(base64 -w 0 -i ./sample_image.jpg)"
    }
  ]
}
END

curl -s -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @sample_request.json \
  http://localhost:8080/predictions/retail-yolo/
