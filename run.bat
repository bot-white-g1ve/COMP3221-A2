@echo off
set dir=F:\uniIssue\Comp3221\A2\COMP3221-A2

rem Start the server
start cmd /k cd "%dir%" ^& python COMP3221_FLServer.py 6000 0
echo We are waiting for the server to finish starting...
timeout /t 5 /nobreak

rem Start the clients
start cmd /k cd "%dir%" ^& python COMP3221_FLClient.py client1 6001 0
start cmd /k cd "%dir%" ^& python COMP3221_FLClient.py client2 6002 0
start cmd /k cd "%dir%" ^& python COMP3221_FLClient.py client3 6003 0
start cmd /k cd "%dir%" ^& python COMP3221_FLClient.py client4 6004 0
start cmd /k cd "%dir%" ^& python COMP3221_FLClient.py client5 6005 0