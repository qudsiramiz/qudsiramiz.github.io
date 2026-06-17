#!/bin/bash

push_changes() {
    local empty_msg="$1"
    local commit_msg="$2"
    local success_msg="$3"

    git add .
    git restore --staged server.pid 2>/dev/null || true

    if git diff --quiet --cached; then
        echo "$empty_msg"
        return 0
    fi

    if ! git commit -m "$commit_msg"; then
        echo "Git commit failed."
        return 1
    fi

    if ! git push origin main; then
        echo "Git push failed."
        return 1
    fi

    echo "$success_msg"
}

# Auto-push all changes before starting
echo "Pushing any recent changes to GitHub..."
if ! push_changes "No new changes to push." "Auto-commit before starting dashboard" "Pushed changes successfully!"; then
    echo "Continuing without pushing startup changes."
fi
echo ""

# Port of our local helper server
PORT=3000

# Start the node server in the background
echo "Starting local helper server on port $PORT..."
node server.js &
SERVER_PID=$!

# Ensure the server is killed when this script exits
cleanup() {
    echo ""
    echo "Stopping helper server..."
    kill "$SERVER_PID" 2>/dev/null
    
    echo "Pushing any final changes to GitHub..."
    if ! push_changes "No final changes to push." "Auto-commit on dashboard shutdown" "Final changes pushed successfully!"; then
        echo "Exiting without pushing final changes."
    fi
    exit
}
trap cleanup EXIT INT TERM

# Wait for server to boot up
sleep 1

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Helper server failed to start. Is port $PORT already in use?"
    exit 1
fi

# Open the local dashboard in the browser via localhost
echo "Opening dashboard in browser via localhost..."
if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:$PORT"
elif command -v sensible-browser > /dev/null; then
    sensible-browser "http://localhost:$PORT"
else
    echo "Could not find a command to open browser automatically."
    echo "Please open http://localhost:$PORT manually in your browser."
fi

# Keep the script running to hold the background server alive and output logs
echo "--------------------------------------------------------"
echo "Helper server is running in background (PID: $SERVER_PID)."
echo "Press Ctrl+C in this terminal to stop the helper server."
echo "--------------------------------------------------------"

# Wait for background process to finish or for trap
wait $SERVER_PID
