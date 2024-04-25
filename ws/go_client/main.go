package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gorilla/websocket"
)

var addr = flag.String("url", "ws://34.91.17.95/ws", "websocket service address")
var image = flag.String("image", "./sample_image.jpg", "image path")

func main() {
	flag.Parse()

	// Reading the binary payload file
	payload, err := os.ReadFile("payload.b")
	if err != nil {
		log.Fatalf("Error reading binary_payload file: %v", err)
	}

	// Connect to WebSocket
	c, _, err := websocket.DefaultDialer.Dial(*addr, nil)
	if err != nil {
		log.Fatalf("Error connecting to WebSocket: %v", err)
	}
	defer c.Close()

	for {
    start := time.Now()
		// Send payload
		err = c.WriteMessage(websocket.BinaryMessage, payload)
		if err != nil {
			log.Fatalf("Error sending message: %v", err)
		}
		fmt.Println("Binary payload sent successfully.")

		// Optionally, read the response if the server sends any
		_, _, err = c.ReadMessage()
		if err != nil {
			log.Fatalf("Error reading server response: %v", err)
		}

    fmt.Println("Time taken: ", time.Since(start))
		// fmt.Printf("Received server response: %s\n", message)
	}
}
