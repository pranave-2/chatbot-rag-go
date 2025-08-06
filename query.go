package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

// Pinecone API key and environment(remove when running both together)
var (
	geminiAPIKey   = os.Getenv("GEMINI_API_KEY")
	pineconeAPIKey = os.Getenv("PINECONE_API_KEY")
	pineconeEnv1   = map[string]string{
		"chatbot-embeddings-384-2x9jann":  "aped-4627-b74a",
		"chatbot-embeddings-512-2x9jann":  "aped-4627-b74a",
		"chatbot-embeddings-1024-2x9jann": "aped-4627-b74a",
	}

	// Three different indexes for different embedding dimensions
	indexes = map[int]string{
		384:  "chatbot-embeddings-384-2x9jann",
		512:  "chatbot-embeddings-512-2x9jann",
		1024: "chatbot-embeddings-1024-2x9jann",
	}
)

type EmbeddingResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

func getEmbedding(text string, dimension int) ([]float32, error) {
	url := "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=" + geminiAPIKey

	payload := map[string]interface{}{
		"content": map[string]interface{}{
			"parts": []map[string]string{
				{"text": text},
			},
		},
		"taskType":             "RETRIEVAL_QUERY", // Use RETRIEVAL_QUERY for semantic search
		"outputDimensionality": dimension,
	}

	body, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %v", err)
	}
	defer res.Body.Close()

	if res.StatusCode != 200 {
		return nil, fmt.Errorf("API returned status %d", res.StatusCode)
	}

	var resp EmbeddingResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return resp.Embedding.Values, nil
}

// Query interface to search similar inputs and get appropriate responses
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

// Search for similar inputs in Pinecone
func searchSimilar(userInput string, dimension int, topK int) (*QueryResult, error) {
	// First get embedding for user input
	embedding, err := getEmbedding(userInput, dimension)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %v", err)
	}

	// Query Pinecone
	indexName := indexes[dimension]
	pineconeEnv := pineconeEnv1[indexName]

	url := fmt.Sprintf("https://%s.svc.%s.pinecone.io/query", indexName, pineconeEnv)

	payload := map[string]interface{}{
		"vector":          embedding,
		"topK":            topK,
		"includeMetadata": true,
		"namespace":       "chatbot-training-data-test-semantic",
	}

	data, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", url, bytes.NewReader(data))
	req.Header.Add("Api-Key", pineconeAPIKey)
	req.Header.Add("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("query failed: %v", err)
	}
	defer res.Body.Close()

	var result QueryResult
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return &result, nil
}

// Generate enhanced response using vector search results
func generateEnhancedResponse(userInput string) {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println(strings.Repeat("=", 60))

	dimensions := []int{384, 512, 1024}

	for _, dim := range dimensions {
		fmt.Printf("\n📊 Dimension %d Results:\n", dim)
		fmt.Println(strings.Repeat("-", 30))

		results, err := searchSimilar(userInput, dim, 3)
		if err != nil {
			fmt.Printf("❌ Error: %v\n", err)
			continue
		}

		if len(results.Matches) == 0 {
			fmt.Println("No matches found")
			continue
		}

		for i, match := range results.Matches {
			fmt.Printf("%d. Score: %.3f\n", i+1, match.Score)
			fmt.Printf("   Similar Input: %s\n", match.Metadata.Input)
			fmt.Printf("   Response: %s\n", match.Metadata.Output)
			fmt.Println()
		}
	}
}

// Test the query functionality
func testQueries() {
	testInputs := []string{
		"I want to book a ride for tomorrow morning",
		"Cancel my pickup for today",
		"What time is my ride tomorrow?",
		"Show me available shifts",
		"Book transport for next week",
	}

	fmt.Println("🧪 Testing Vector Search Functionality...")

	for _, input := range testInputs {
		generateEnhancedResponse(input)
		fmt.Println("\n" + strings.Repeat("=", 80) + "\n")
	}
}

// Main function for query testing
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

	if len(os.Args) > 1 && os.Args[1] == "test" {
		testQueries()
		return
	}

	// Interactive mode
	fmt.Println("🤖 Chatbot Vector Search - Interactive Mode")
	fmt.Println("Type your queries (or 'quit' to exit):")

	var input string

	fmt.Print("\n> ")
	fmt.Scanln(&input)

	generateEnhancedResponse(input)

	fmt.Println("👋 Goodbye!")
}
