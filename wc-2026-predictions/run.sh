#!/bin/bash

# Auto-push all changes before starting
echo "Pushing any recent changes to GitHub..."
git add .
if git diff --quiet --cached; then
    echo "No new changes to push."
else
    git commit -m "Auto-commit before starting dashboard"
    git push origin main
    echo "Pushed changes successfully!"
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
    kill $SERVER_PID 2>/dev/null
    
    echo "Pushing any final changes to GitHub..."
    git add .
    if git diff --quiet --cached; then
        echo "No final changes to push."
    else
        git commit -m "Auto-commit on dashboard shutdown"
        git push origin main
        echo "Final changes pushed successfully!"
    fi
    exit
}
trap cleanup EXIT INT TERM

# Wait for server to boot up
sleep 1

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
