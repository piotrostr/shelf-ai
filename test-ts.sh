#!/bin/bash

# check if works for posting files
res=$(curl -s -X POST http://localhost:8080/predictions/retail-yolo -T ./sample_image.jpg | jq length)

# check if non-empty response
if [ "$res" -eq 0 ]; then
  echo "Test failed for posting files"
  exit 1
else 
  echo "Test passed for posting files, got $res objects"
fi

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

res=$(curl -s -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @sample_request.json \
  http://localhost:8080/predictions/retail-yolo/)

res_len=$(echo $res | jq length)

if [ "$res_len" -eq 0 ]; then
  echo "Test failed for JSON with base64 encoded image"
  exit 1
else
  echo "Test passed for JSON with base64 encoded image, got $res_len objects"
fi
