
import pandas as pd, os

class DuplicateTracker:
 """In-memory duplicate tracker. Loads existing entries once,
 then checks O(1) via a set instead of re-reading CSV every time."""

 def __init__(self, path):
  self._seen = set()
  if os.path.exists(path):
   df = pd.read_csv(path, usecols=['query_id','run_number','machine_name'])
   for _, row in df.iterrows():
    self._seen.add((row['query_id'], row['run_number'], row['machine_name']))

 def exists(self, q, r, m):
  return (q, r, m) in self._seen

 def add(self, q, r, m):
  self._seen.add((q, r, m))
