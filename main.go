package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/joho/godotenv"
)

// Configuration
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

type InputOutputPair struct {
	Input  string
	Output string
}

type Vector struct {
	ID       string                 `json:"id"`
	Values   []float32              `json:"values"`
	Metadata map[string]interface{} `json:"metadata"`
}

type EmbeddingResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

// Extract input-output pairs from the documentation
func extractInputOutputPairs(filename string) ([]InputOutputPair, error) {
	if filename != "" {
		if data, err := os.ReadFile(filename); err == nil {
			var pairs []InputOutputPair
			if err := json.Unmarshal(data, &pairs); err != nil {
				return nil, fmt.Errorf("failed to parse JSON file %s: %v", filename, err)
			}
			fmt.Printf("📁 Loaded %d pairs from %s\n", len(pairs), filename)
			return pairs, nil
		}
	}

	pairs := []InputOutputPair{
		// Booking scenarios
		{"Book transport for tomorrow at 8 AM", "Got it! You're scheduling a pickup for tomorrow at 8 AM. Can you confirm your drop location is your office?"},
		{"I want pickup from home at 7:30 AM on Monday", "Perfect! I'm booking your pickup for Monday at 7:30 AM from your home address. Your roster is confirmed! You will receive driver details 30 minutes before the trip."},
		{"Schedule my pickup for 6 PM today", "I've scheduled your pickup for today at 6 PM. Your booking is confirmed and you'll receive driver details shortly."},
		{"Add me to the transport list for tomorrow's night shift", "I've added you to the transport roster for tomorrow's night shift. You'll receive confirmation with driver details 30 minutes before your trip."},

		// Viewing schedule scenarios
		{"Show me my roster for this week", "Here's your upcoming roster:\n• Tomorrow - Pickup at 7:30 AM, Drop at 6 PM\n• Wednesday - Pickup at 8 AM\n• Friday - No Roster"},
		{"Do I have a trip scheduled for tomorrow?", "You have a pickup scheduled tomorrow at 8 AM from your home address."},
		{"What time is my pickup today?", "You have a pickup scheduled today at 6 PM from your home address."},
		{"Show my upcoming transport schedule", "Here are your upcoming trips:\n• Today - Drop at 6 PM\n• Tomorrow - Pickup at 8 AM\n• Thursday - Pickup at 7:30 AM, Drop at 6:30 PM"},

		// Modification scenarios
		{"Change my pickup time to 9 AM tomorrow", "I found your roster for tomorrow at 8 AM. I've updated your pickup time to 9 AM. You'll receive updated trip details shortly."},
		{"Reschedule my drop to 7 PM instead of 6", "Your drop-off time has been updated to 7 PM. Updated trip details will be shared with you."},
		{"Edit the time for Monday roster", "You have a roster for Monday at 6 PM. What time would you like to change it to?"},

		// Cancellation scenarios
		{"Cancel my transport for tomorrow", "I found a roster for tomorrow at 8 AM. Your transport for tomorrow has been cancelled successfully."},
		{"I won't need pickup on Friday", "Your pickup for Friday has been cancelled."},
		{"I'm working from home tomorrow, cancel the ride", "Done! Your ride for tomorrow has been successfully cancelled."},

		// Help and guidance
		{"How do I book a trip?", "I can help you book a transport! Just tell me the date and time. For example: 'Book a pickup for tomorrow at 9 AM' and I'll handle the rest."},
		{"Can I cancel a roster?", "Yes, you can cancel a roster anytime! Just tell me which trip you want to cancel, like 'Cancel my ride for tomorrow' and I'll take care of it."},
		{"What is a roster?", "A roster is your scheduled ride for a specific shift or date. It includes pickup and drop-off times and locations."},

		// Available shifts queries
		{"What are the available login shifts for tomorrow?", "Your login shifts for tomorrow are every 30 minutes starting from 7 AM."},
		{"Show me logout shifts for today", "Logout shifts for today are every 30 minutes starting from 6 PM."},
		{"What shifts are available after 8 AM tomorrow?", "Login shifts are available every 30 minutes starting from 8:10 AM tomorrow."},

		// Multi-day booking
		{"Book my rides for the entire week", "I can help you book rides for multiple days. What are the start and end dates for your weekly booking?"},
		{"I need transport from Monday to Friday", "I'll book your transport from Monday to Friday. What time would you prefer for your daily rides?"},

		// Error handling scenarios
		{"Book", "Can you please specify the date and time for your trip?"},
		{"I need it", "Did you mean to book a ride? Please provide the date and time, like 'Pickup at 7 AM tomorrow'."},
		{"Next week", "I'd be happy to help with next week's transport. Could you specify the exact dates and times you need?"},

		// Real-time updates
		{"Where is my cab?", "Your cab (KA01AB1234) is currently 5 minutes away from your pickup location. Driver Ramesh will contact you when he arrives."},
		{"Is my driver here?", "Your cab (KA01AB1234) has arrived at your location. Driver Ramesh is waiting at the pickup point."},

		// System prompt derived patterns
		{"book my ride for tomorrow", "What time would you like your ride for tomorrow?"},
		{"yes", "What time works for you?"},
		{"cancel my ride for tomorrow", "Your ride for tomorrow has been cancelled successfully."},
		{"did I book a ride for Monday?", "Let me check your bookings for Monday..."},
		{"book multiple days", "What are the start and end dates for your multi-day booking?"},
	}

	return pairs, nil
}

// Get embedding from Gemini API
func getEmbedding(text string, dimension int) ([]float32, error) {
	url := "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=" + geminiAPIKey

	payload := map[string]interface{}{
		"content": map[string]interface{}{
			"parts": []map[string]string{
				{"text": text},
			},
		},
		"taskType":             "RETRIEVAL_DOCUMENT", // Use RETRIEVAL_QUERY for semantic search
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

// Upload vectors to specific Pinecone index
func upsertToPinecone(vectors []Vector, dimension int) error {
	indexName := indexes[dimension]
	pineconeEnv := pineconeEnv1[indexName]
	url := fmt.Sprintf("https://%s.svc.%s.pinecone.io/vectors/upsert", indexName, pineconeEnv)

	payload := map[string]interface{}{
		"vectors":   vectors,
		"namespace": "chatbot-training-data-test-semantic",
	}
	data, _ := json.Marshal(payload)

	req, _ := http.NewRequest("POST", url, bytes.NewReader(data))
	req.Header.Add("Api-Key", pineconeAPIKey)
	req.Header.Add("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to upload to Pinecone: %v", err)
	}
	defer res.Body.Close()

	fmt.Printf("✅ Pinecone upload to %s (dim %d): %s\n", indexName, dimension, res.Status)

	if res.StatusCode >= 400 {
		var errBody bytes.Buffer
		errBody.ReadFrom(res.Body)
		return fmt.Errorf("Pinecone error %d: %s", res.StatusCode, errBody.String())
	}

	return nil
}

// Process and upload data for all dimensions
func processAndUpload() {
	pairs, _ := extractInputOutputPairs("test_embedding.json")
	dimensions := []int{384, 512, 1024}

	fmt.Printf("📊 Processing %d input-output pairs for %d different dimensions...\n", len(pairs), len(dimensions))

	for _, dim := range dimensions {
		fmt.Printf("\n🔄 Processing dimension %d...\n", dim)
		var vectors []Vector

		for i, pair := range pairs {
			// Get embedding for the input
			embedding, err := getEmbedding(pair.Input, dim)
			if err != nil {
				fmt.Printf("❌ Error getting embedding for pair %d: %v\n", i, err)
				continue
			}

			// Create vector with rich metadata
			vector := Vector{
				ID:     fmt.Sprintf("pair_%d_dim_%d", i, dim),
				Values: embedding,
				Metadata: map[string]interface{}{
					"input":      pair.Input,
					"output":     pair.Output,
					"dimension":  dim,
					"pair_id":    i,
					"created_at": time.Now().Unix(),
					"input_len":  len(pair.Input),
					"output_len": len(pair.Output),
				},
			}

			vectors = append(vectors, vector)

			// Rate limiting - Gemini has rate limits
			time.Sleep(100 * time.Millisecond)
			if (i+1)%10 == 0 {
				fmt.Printf("   📝 Processed %d/%d pairs for dim %d\n", i+1, len(pairs), dim)
			}
		}

		// Upload to Pinecone
		if len(vectors) > 0 {
			err := upsertToPinecone(vectors, dim)
			if err != nil {
				fmt.Printf("❌ Failed to upload dim %d: %v\n", dim, err)
			} else {
				fmt.Printf("✅ Successfully uploaded %d vectors for dimension %d\n", len(vectors), dim)
			}
		}

		// Small delay between dimensions
		time.Sleep(500 * time.Millisecond)
	}
}

// Utility function to save logs
func saveProcessingLogs(pairs []InputOutputPair) {
	filename := fmt.Sprintf("output_logs/processing_log_%d.txt", time.Now().Unix())
	f, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Failed to create log file: %v\n", err)
		return
	}
	defer f.Close()

	f.WriteString(fmt.Sprintf("Processing Log - %s\n", time.Now().Format("2006-01-02 15:04:05")))
	f.WriteString(fmt.Sprintf("Total pairs processed: %d\n", len(pairs)))
	f.WriteString(fmt.Sprintf("Dimensions: 384, 512, 1024\n\n"))

	for i, pair := range pairs {
		f.WriteString(fmt.Sprintf("Pair %d:\n", i+1))
		f.WriteString(fmt.Sprintf("Input: %s\n", pair.Input))
		f.WriteString(fmt.Sprintf("Output: %s\n\n", pair.Output))
	}

	fmt.Printf("📄 Processing log saved to: %s\n", filename)
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

	fmt.Println("🚀 Starting Chatbot Vector Database Setup...")
	fmt.Printf("📋 Target indexes: %v\n", indexes)

	// Create output directory
	os.MkdirAll("output_logs", 0755)

	// Extract and save processing logs
	pairs, _ := extractInputOutputPairs("extracted_input_output_pairs.json")
	saveProcessingLogs(pairs)

	// Process and upload all data
	processAndUpload()

	fmt.Println("\n🎉 Vector database setup complete!")
	fmt.Println("💡 Your chatbot now has enhanced context from input-output pairs stored in Pinecone.")

}
