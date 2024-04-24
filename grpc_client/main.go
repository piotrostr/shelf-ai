package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"runtime/trace"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/types/known/structpb"
)

var (
	useRawPredict = flag.Bool("raw", false, "use raw predict (gax fallback)")
	useHTTP       = flag.Bool("http", false, "use HTTP 1.1")
	usegRPC       = flag.Bool("grpc", false, "use gRPC")
	runTrace      = flag.Bool("trace", false, "run trace")
)

type Instance struct {
	Content string `json:"content"`
}

type Payload struct {
	Instances []Instance `json:"instances"`
}

func getPayload(encodedImage string) *httpbody.HttpBody {
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
	return body
}

func main() {
	flag.Parse()
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

	sampleImagePath := "../sample_image_640x360.jpg"

	log.Println("reading and encoding image" + sampleImagePath)

	imageData, err := os.ReadFile(sampleImagePath)
	if err != nil {
		log.Fatal(err)
	}

	if *runTrace {
		f, err := os.Create("trace.out")
		if err != nil {
			log.Fatal(err)
		}
		err = trace.Start(f)
		defer trace.Stop()
	}

	encodedImage := base64.StdEncoding.EncodeToString(imageData)

	numReqs := 100

	log.Printf("sending %d PredictRequests\n", numReqs)

	endpoint := "projects/461501354025/locations/europe-west4/endpoints/4809408995427090432"

	if *useHTTP {
		log.Println("using HTTP 1.1")
		urlBase := "https://europe-west4-aiplatform.googleapis.com/v1/"
		body := bytes.NewReader(getPayload(encodedImage).Data)
		url := urlBase + endpoint + ":predict"
		req, err := http.NewRequest("POST", url, body)
		if err != nil {
			log.Fatal(err)
		}
		// Read the credentials from the service account file
		creds, err := os.ReadFile("creds.json")
		if err != nil {
			log.Fatalf("Failed to read service account file: %v", err)
		}

		// Get the Google credentials from the service account file
		config, err := google.JWTConfigFromJSON(creds, "https://www.googleapis.com/auth/cloud-platform")
		if err != nil {
			log.Fatalf("JWTConfigFromJSON failed: %v", err)
		}

		// Create a new HTTP client using the token source
		client := config.Client(ctx)
		if err != nil {
			log.Fatalf("Failed to create ID token client: %v", err)
		}
		req.Header.Add("accept", "application/json")

		for i := 0; i < numReqs; i++ {
			start := time.Now()
			resp, err := client.Do(req)
			if err != nil {
				log.Fatal(err)
			}
			defer resp.Body.Close()
			t := time.Now()
			elapsed := t.Sub(start)
			log.Printf("took %s\n", elapsed)

			// Read the response body
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				log.Fatalf("Failed to read response body: %v", err)
			}

			fmt.Printf("Response: %s\n", responseBody)
		}
		return
	}

	if *useRawPredict {
		log.Println("using RawPredict (HTTP 1.1 gax)")

		for i := 0; i < numReqs; i++ {
			req := &aiplatformpb.RawPredictRequest{
				Endpoint: endpoint,
				HttpBody: getPayload(encodedImage),
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

	if *usegRPC {
		log.Println("using Predict (gRPC)")
		for i := 0; i < numReqs; i++ {
			instance, err := structpb.NewValue(map[string]interface{}{
				"content": encodedImage,
			})
			if err != nil {
				log.Fatal(err)
			}
			req := &aiplatformpb.PredictRequest{
				Endpoint: endpoint,
				Instances: []*structpb.Value{
					instance,
				},
			}

			start := time.Now()
			_, err = c.Predict(ctx, req)
			t := time.Now()
			elapsed := t.Sub(start)
			log.Printf("took %s\n", elapsed)
			if err != nil {
				log.Fatal(err)
			}
		}
	}
}
