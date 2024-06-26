#!/bin/bash 

dir="/Users/advancedai/Desktop/UniIssue/comp3221/A2/COMP3221-A2"

# Start the server in a new Terminal window
osascript -e "tell application \"Terminal\" to do script \"cd '$dir'; python3 COMP3221_FLServer.py 6000 0\""
echo "We are waiting for the server to finish starting..."
sleep 5 # Sleep and wait for the server to finish starting

# List of sleep durations for each client
sleep_durations=(1 1 1 1)  # Array starts at index 0

# Loop to start each client in a new Terminal window
for i in {1..5}
do
  client_index=$((i - 1))  # Convert 1-based index to 0-based for the array
  echo "Starting client $i..."
  osascript -e "tell application \"Terminal\" to do script \"cd '$dir'; python3 COMP3221_FLClient.py client$i $((6000 + i)) 0\""
  sleep ${sleep_durations[$client_index]}  # Use adjusted index to access correct sleep duration
done
