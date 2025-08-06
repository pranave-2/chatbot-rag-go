package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
)

var (
	geminiAPIKey   = os.Getenv("GEMINI_API_KEY")
	pineconeAPIKey = os.Getenv("PINECONE_API_KEY")
	pineconeEnv1   = map[string]string{
		"chatbot-embeddings-384-2x9jann":  "aped-4627-b74a",
		"chatbot-embeddings-512-2x9jann":  "aped-4627-b74a",
		"chatbot-embeddings-1024-2x9jann": "aped-4627-b74a",
	}
	indexes = map[int]string{
		384:  "chatbot-embeddings-384-2x9jann",
		512:  "chatbot-embeddings-512-2x9jann",
		1024: "chatbot-embeddings-1024-2x9jann",
	}
)

type QueryResult struct {
	Matches []struct {
		ID       string  `json:"id"`
		Score    float32 `json:"score"`
		Metadata struct {
			Input     string `json:"input"`
			Output    string `json:"output"`
			Dimension int    `json:"dimension"`
		} `json:"metadata"`
	} `json:"matches"`
}

func diagnoseIndex(dimension int) error {
	indexName := indexes[dimension]
	pineconeEnv := pineconeEnv1[indexName]
	url := fmt.Sprintf("https://%s.svc.%s.pinecone.io/query", indexName, pineconeEnv)

	fmt.Printf("\n🔍 Checking index: %s (%dD)\n", indexName, dimension)
	fmt.Println("----------------------------------------------------------")

	// Send a zero-vector to retrieve everything
	zeroVector := make([]float32, dimension)

	payload := map[string]interface{}{
		"vector":          zeroVector,
		"topK":            100,
		"includeMetadata": true,
		"namespace":       "chatbot-training-data",
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Api-Key", pineconeAPIKey)
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("❌ failed to query index: %v", err)
	}
	defer res.Body.Close()

	if res.StatusCode != 200 {
		return fmt.Errorf("❌ Pinecone API returned status %d", res.StatusCode)
	}

	var result QueryResult
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		return fmt.Errorf("❌ failed to decode response: %v", err)
	}

	if len(result.Matches) == 0 {
		fmt.Println("⚠️ No vectors found.")
		return nil
	}

	for i, m := range result.Matches {
		fmt.Printf("%2d. Score: %.3f | Input: %q\n", i+1, m.Score, m.Metadata.Input)
		if m.Metadata.Output == "" || m.Metadata.Input == "" {
			fmt.Println("   ⚠️ WARNING: Missing input/output in metadata")
		}
		if m.Metadata.Input == m.Metadata.Output {
			fmt.Println("   ⚠️ Suspicious: Input and Output are same")
		}
		if m.Metadata.Input == "Similar Input: System Response:" {
			fmt.Println("   ❌ Corrupted: Looks like concatenated string")
		}
		if i >= 20 {
			fmt.Printf("   ...only showing first 20 of %d vectors\n", len(result.Matches))
			break
		}
	}

	return nil
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}
	geminiAPIKey = os.Getenv("GEMINI_API_KEY")
	pineconeAPIKey = os.Getenv("PINECONE_API_KEY")
	if geminiAPIKey == "" {
		fmt.Println("❌ GEMINI_API_KEY not set")
		return
	}
	if pineconeAPIKey == "" {
		fmt.Println("❌ PINECONE_API_KEY not set")
		return
	}

	fmt.Println("🧠 Debugging Pinecone Vector Data for Issues")
	fmt.Println("============================================")

	for _, dim := range []int{384, 512, 1024} {
		if err := diagnoseIndex(dim); err != nil {
			fmt.Printf("❌ Error with %dD index: %v\n", dim, err)
		}
	}
}
