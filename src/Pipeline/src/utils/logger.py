
from datetime import datetime
def log(msg):
 with open('logs/pipeline.log','a') as f:
  f.write(f"{datetime.now()} {msg}\n")
