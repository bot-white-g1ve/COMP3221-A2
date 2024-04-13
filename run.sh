dir="/Users/advancedai/Desktop/UniIssue/comp3221/A2/COMP3221-A2"

osascript -e 'tell application "Terminal" to do script "cd '"$dir"';python3 COMP3221_FLServer.py 6000 0"'
# sleep and waiting for the server to finish starting
echo "We are waiting for the server to finish starting..."
sleep 5
osascript -e 'tell application "Terminal" to do script "cd '"$dir"';python3 COMP3221_FLClient.py client1 6001 0"'
osascript -e 'tell application "Terminal" to do script "cd '"$dir"';python3 COMP3221_FLClient.py client2 6002 0"'