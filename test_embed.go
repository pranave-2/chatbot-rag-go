package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		fmt.Println("❌ Missing GEMINI_API_KEY")
		return
	}

	url := "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=" + apiKey
	payload := map[string]interface{}{
		"content": map[string]interface{}{
			"parts": []map[string]string{
				{"text": "Book my ride for tomorrow"},
			},
		},
		"taskType": "RETRIEVAL_DOCUMENT", // this matters!
	}

	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(body))
	if err != nil {
		fmt.Printf("❌ Request error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	fmt.Printf("🔁 Gemini Response:\n%s\n", string(respBody))
}
