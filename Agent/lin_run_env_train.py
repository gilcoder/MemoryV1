import os
import time

input_port = 7070
output_port = 8080
host = "127.0.0.1"
sep = os.sep
APP_PATH = "#TYPEAPPPATHHERE"
for i in range(8):
    path = "%s --ai4u_inputport %d --ai4u_outputport %d --ai4u_remoteip %s --ai4u_timescale 1 --ai4u_targetframerate 1000 --ai4u_vsynccount 0 &"%(APP_PATH, input_port + i, output_port + i, host)
    os.system(path)
    time.sleep(1)
