package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type responseBody struct {
	Feedback   string `json:"feedback"`
	ReturnCode int    `json:"returncode"`
	Detail     any    `json:"detail"`
}

type exchangeResponse struct {
	StatusCode     int             `json:"status_code"`
	Body           json.RawMessage `json:"body"`
	TransportError string          `json:"transport_error"`
}

func fail(format string, values ...any) {
	fmt.Fprintf(os.Stderr, "PTXBENCH_INFRA_ERROR: "+format+"\n", values...)
	os.Exit(2)
}

func actionID() string {
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		fail("cannot generate action id: %v", err)
	}
	value[6] = (value[6] & 0x0f) | 0x40
	value[8] = (value[8] & 0x3f) | 0x80
	encoded := hex.EncodeToString(value)
	return encoded[0:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" + encoded[16:20] + "-" + encoded[20:]
}

func validatedWorkload(data []byte) (json.RawMessage, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, fmt.Errorf("workload JSON must be an object")
	}
	return json.RawMessage(data), nil
}

func printUsage(output io.Writer) {
	fmt.Fprintln(output, "usage: ptxbench-eval [-h] kernel workload")
	fmt.Fprintln(output)
	fmt.Fprintln(output, "Synchronously retrieve controlled PTXBench feedback.")
	fmt.Fprintln(output)
	fmt.Fprintln(output, "positional arguments:")
	fmt.Fprintln(output, "  kernel")
	fmt.Fprintln(output, "  workload")
	fmt.Fprintln(output)
	fmt.Fprintln(output, "options:")
	fmt.Fprintln(output, "  -h, --help  show this help message and exit")
}

func exchangeEvaluate(exchangeDir string, payload []byte, action string, timeout time.Duration) (int, []byte, error) {
	if err := os.MkdirAll(exchangeDir, 0700); err != nil {
		return 0, nil, fmt.Errorf("cannot create evaluator exchange: %w", err)
	}
	key := fmt.Sprintf("%x", sha256.Sum256([]byte(action)))
	requestPath := filepath.Join(exchangeDir, key+".request.json")
	responsePath := filepath.Join(exchangeDir, key+".response.json")
	envelope, err := json.Marshal(map[string]any{
		"schema_version":  1,
		"timeout_seconds": timeout.Seconds(),
		"request":         json.RawMessage(payload),
	})
	if err != nil {
		return 0, nil, fmt.Errorf("cannot encode evaluator exchange request: %w", err)
	}
	temporary, err := os.CreateTemp(exchangeDir, ".request-*")
	if err != nil {
		return 0, nil, fmt.Errorf("cannot create evaluator exchange request: %w", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if _, err := temporary.Write(append(envelope, '\n')); err != nil {
		temporary.Close()
		return 0, nil, fmt.Errorf("cannot write evaluator exchange request: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return 0, nil, fmt.Errorf("cannot sync evaluator exchange request: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return 0, nil, fmt.Errorf("cannot close evaluator exchange request: %w", err)
	}
	if err := os.Rename(temporaryPath, requestPath); err != nil {
		return 0, nil, fmt.Errorf("cannot publish evaluator exchange request: %w", err)
	}
	defer os.Remove(requestPath)
	defer os.Remove(responsePath)

	deadline := time.NewTimer(timeout + 10*time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		encoded, readErr := os.ReadFile(responsePath)
		if readErr == nil {
			var result exchangeResponse
			if err := json.Unmarshal(encoded, &result); err != nil {
				return 0, nil, fmt.Errorf("invalid evaluator exchange response: %w", err)
			}
			if result.TransportError != "" {
				return 0, nil, fmt.Errorf("%s", result.TransportError)
			}
			if result.StatusCode == 0 {
				return 0, nil, fmt.Errorf("evaluator exchange response has no HTTP status")
			}
			return result.StatusCode, []byte(result.Body), nil
		}
		if !os.IsNotExist(readErr) {
			return 0, nil, fmt.Errorf("cannot read evaluator exchange response: %w", readErr)
		}
		select {
		case <-ticker.C:
		case <-deadline.C:
			return 0, nil, fmt.Errorf("timed out waiting for evaluator exchange response")
		}
	}
}

func main() {
	var explicitAction string
	var jsonOutput bool
	var timeoutSeconds float64
	flag.Usage = func() {
		printUsage(flag.CommandLine.Output())
	}
	flag.StringVar(&explicitAction, "action-id", "", "stable idempotency key")
	flag.BoolVar(&jsonOutput, "json", false, "print the complete JSON response")
	flag.Float64Var(&timeoutSeconds, "timeout", 900, "request timeout in seconds")
	flag.Parse()
	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: ptxbench-eval [-h] kernel workload")
		os.Exit(2)
	}

	gateway := strings.TrimRight(os.Getenv("PTXBENCH_GATEWAY_URL"), "/")
	exchangeDir := os.Getenv("PTXBENCH_EVAL_EXCHANGE_DIR")
	runID := os.Getenv("PTXBENCH_RUN_ID")
	token := os.Getenv("PTXBENCH_RUN_TOKEN")
	if runID == "" || (exchangeDir == "" && (gateway == "" || token == "")) {
		fail("PTXBENCH_GATEWAY_URL, PTXBENCH_RUN_ID, and PTXBENCH_RUN_TOKEN are required")
	}
	source, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		fail("cannot read kernel: %v", err)
	}
	workloadBytes, err := os.ReadFile(flag.Arg(1))
	if err != nil {
		fail("cannot read workload: %v", err)
	}
	workload, err := validatedWorkload(workloadBytes)
	if err != nil {
		fail("invalid workload JSON: %v", err)
	}
	if explicitAction == "" {
		explicitAction = actionID()
	}
	payload, err := json.Marshal(map[string]any{
		"schema_version": 1,
		"run_id":         runID,
		"action_id":      explicitAction,
		"source":         string(source),
		"workload":       workload,
	})
	if err != nil {
		fail("cannot encode request: %v", err)
	}

	timeout := time.Duration(timeoutSeconds * float64(time.Second))
	var statusCode int
	var body []byte
	if exchangeDir != "" {
		statusCode, body, err = exchangeEvaluate(exchangeDir, payload, explicitAction, timeout)
		if err != nil {
			fail("%v", err)
		}
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, gateway+"/v1/evaluate", bytes.NewReader(payload))
		if err != nil {
			fail("cannot create request: %v", err)
		}
		request.Header.Set("Authorization", "Bearer "+token)
		request.Header.Set("Content-Type", "application/json")
		response, err := (&http.Client{}).Do(request)
		if err != nil {
			fail("%v", err)
		}
		defer response.Body.Close()
		statusCode = response.StatusCode
		body, err = io.ReadAll(io.LimitReader(response.Body, 64<<20))
		if err != nil {
			fail("cannot read response: %v", err)
		}
	}
	var decoded responseBody
	if err := json.Unmarshal(body, &decoded); err != nil {
		fail("HTTP %d returned invalid JSON: %s", statusCode, string(body))
	}
	if statusCode >= 400 {
		if decoded.Detail != nil {
			encoded, _ := json.Marshal(decoded.Detail)
			fail("HTTP %d: %s", statusCode, string(encoded))
		}
		fail("HTTP %d: %s", statusCode, string(body))
	}
	if jsonOutput {
		var pretty bytes.Buffer
		if err := json.Indent(&pretty, body, "", "  "); err != nil {
			fail("cannot format response: %v", err)
		}
		fmt.Println(pretty.String())
	} else {
		fmt.Println(decoded.Feedback)
	}
	os.Exit(decoded.ReturnCode)
}
