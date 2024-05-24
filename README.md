Requirements:
Python 3.11.7
pandas
torch
numpy
matplotlib

How to run the program?
If using windows: In run.bat, modify the variable "dir" to the root
directory of the program. Then run run.bat
If using Linux: In run.sh, modify the variable "dir" to the root directory
of the program. Then bash run.sh

How to reproduce the experimental results?
evaluate on learning rate: modify the variable "lr" (at the top) in
COMP3221_FLClient.py
evaluate on epochs: modify the variable "epochs" (at the top) in
COMP3221_FLClient.py
evaluate on mini-batch: modify the variable "mini_batch_size" (at the top)
in COMP3221_FLClient.py. Also when python COMP3221_FLClient.py, set the
parameter <Opt-Method> to 1 (modify run.sh or run.bat)
evaluate on subsampling: When python COMP3221_FLServer.py, set the
parameter <Sub-Client> to 0-4 (modify run.sh or run.bat)
