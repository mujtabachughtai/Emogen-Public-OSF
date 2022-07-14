import numpy as np
from numpy import genfromtxt
import subprocess
import sys

csv_file = sys.argv[1] # /home/emogen/emogenWeb/shared/public/111/1655833333538/HAPPY/MALE/initialisation.csv
save_dir = sys.argv[2] # /home/emogen/emoGen_sandbox/EmoGen_deployment_v2/Deployment_version/user_output/initialisation_transposed.csv
csv_data = genfromtxt(csv_file, delimiter=',')

np.savetxt(save_dir, np.transpose(csv_data), delimiter=",")

subprocess.run(["/home/emogen/emoGen_sandbox/EmoGen_deployment_v2/Deployment_version/run_visualisation.sh"])
