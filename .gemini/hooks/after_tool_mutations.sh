#!/usr/bin/env bash
# Intercepts write_file and replace to enforce formatting and typing

# Read the JSON from stdin into a variable
INPUT=$(cat)

# Extract tool_name and file_path using python for robust JSON parsing
TOOL_NAME=$(echo "$INPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('tool_name', ''))")
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); params=data.get('tool_parameters', {}); print(params.get('file_path', ''))")

# We only care about mutative file operations
if [[ "$TOOL_NAME" != "write_file" && "$TOOL_NAME" != "replace" ]]; then
    echo '{"decision": "allow"}'
    exit 0
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Verify based on file extension
VERIFY_SCRIPT=""
if [[ "$FILE_PATH" == *.svelte || "$FILE_PATH" == *.ts || "$FILE_PATH" == *.js ]]; then
    VERIFY_SCRIPT="$DIR/.agents/scripts/verify_frontend.sh"
elif [[ "$FILE_PATH" == *.py ]]; then
    VERIFY_SCRIPT="$DIR/.agents/scripts/verify_backend.sh"
fi

if [[ -n "$VERIFY_SCRIPT" && -x "$VERIFY_SCRIPT" ]]; then
    # Run the verification, capturing stderr output which holds the details
    VERIFY_OUT=$("$VERIFY_SCRIPT" 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        # Escape the output for JSON encoding
        ESCAPED_OUT=$(echo "$VERIFY_OUT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
        
        # Deny the operation, providing the output back to the agent for self-correction
        echo "{\"decision\": \"deny\", \"reason\": \"Verification failed: ${ESCAPED_OUT}\"}"
        exit 0
    fi
fi

echo '{"decision": "allow"}'
exit 0