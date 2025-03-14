import time
import subprocess

# Wartezeit in Sekunden (2 Stunden = 7200 Sekunden)
time.sleep(10000)

# Terminal-Befehl ausführen (hier als Beispiel 'echo "Hallo Welt"')
subprocess.run("CUDA_VISIBLE_DEVICES=1 wandb agent belaschindler-university-hamburg/Thesis_Experimente-LayUp_sweeps_question1/6x2llpw9", shell=True)
