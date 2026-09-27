package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestValidatedWorkloadPreservesJSONNumberSpelling(t *testing.T) {
	input := []byte(`{"lower_bound":-5.0,"scale":0.08838834764831843,"count":5}`)
	workload, err := validatedWorkload(input)
	if err != nil {
		t.Fatalf("validatedWorkload returned an error: %v", err)
	}

	payload, err := json.Marshal(map[string]any{"workload": workload})
	if err != nil {
		t.Fatalf("cannot marshal request payload: %v", err)
	}
	if !bytes.Contains(payload, []byte(`"lower_bound":-5.0`)) {
		t.Fatalf("integral float spelling was not preserved: %s", payload)
	}
	if !bytes.Contains(payload, []byte(`"scale":0.08838834764831843`)) {
		t.Fatalf("non-integral float spelling was not preserved: %s", payload)
	}
}

func TestValidatedWorkloadRequiresJSONObject(t *testing.T) {
	for _, input := range [][]byte{[]byte(`null`), []byte(`[]`)} {
		if _, err := validatedWorkload(input); err == nil {
			t.Fatalf("validatedWorkload accepted non-object JSON: %s", input)
		}
	}
}

func TestHelpOnlyExposesAgentSubmissionArguments(t *testing.T) {
	var output bytes.Buffer
	printUsage(&output)
	helpText := output.String()

	if !strings.Contains(helpText, "usage: ptxbench-eval [-h] kernel workload") {
		t.Fatalf("help is missing the agent-facing usage: %q", helpText)
	}
	for _, hiddenOption := range []string{
		"--action-id",
		"--json",
		"--gateway-url",
		"--run-id",
		"--run-token",
		"--timeout",
	} {
		if strings.Contains(helpText, hiddenOption) {
			t.Errorf("help exposes internal option %q", hiddenOption)
		}
	}
}
