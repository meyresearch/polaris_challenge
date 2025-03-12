import subprocess
import requests
from datetime import datetime
import os

TOKEN = os.getenv("TELEGRAM_TOKEN")
print(TOKEN)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
def notify_end(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    MESSAGE = f"[{timestamp}] {message}"
    requests.get(url, params={"chat_id": CHAT_ID, "text": MESSAGE})

    
def run_boltz():
    notify_end("test Boltz predict started")
    file = "train_fasta_subset"
    command = ["boltz", "predict", file , "--use_msa_server"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        msg = "Boltz predict ran successfully"
        print(msg)
        notify_end(msg)
        
    else:
        msg = "Error running boltz predict "
        err = result.stderr
        print(result.stderr)
        notify_end(msg+err)
        
        
if __name__ == "__main__":
    run_boltz()