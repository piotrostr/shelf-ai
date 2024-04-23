package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log"
	"os"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genproto/googleapis/api/httpbody"
)

type Instance struct {
	Content string `json:"content"`
}

type Payload struct {
	Instances []Instance `json:"instances"`
}

func main() {
	ctx := context.Background()
	c, err := aiplatform.NewPredictionClient(
		ctx,
		option.WithEndpoint("europe-west4-aiplatform.googleapis.com:443"),
		option.WithCredentialsFile("./creds.json"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	log.Println("client instantiated")

	sampleImagePath := "./sample_image.jpg"

	log.Println("reading and encoding image" + sampleImagePath)

	imageData, err := os.ReadFile(sampleImagePath)
	if err != nil {
		log.Fatal(err)
	}

	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	payload := Payload{
		Instances: []Instance{
			{
				Content: encodedImage,
			},
		},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	body := &httpbody.HttpBody{}
	body.Data = payloadBytes

	log.Println("sending 10 PredictRequest")

	for i := 0; i < 10; i++ {
		req := &aiplatformpb.RawPredictRequest{
			Endpoint: "projects/461501354025/locations/europe-west4/endpoints/4809408995427090432",
			HttpBody: body,
		}

		start := time.Now()
		_, err := c.RawPredict(ctx, req)
		t := time.Now()
		elapsed := t.Sub(start)
		log.Printf("took %s\n", elapsed)
		if err != nil {
			log.Fatal(err)
		}
	}
}
