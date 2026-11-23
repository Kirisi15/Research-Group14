
import json,time,socket,pandas as pd,glob,os
from datetime import datetime
from src.database.connection import get_connection
from src.features.semantic_features import extract as sqlf
from src.features.system_features import extract as sysf
from src.features.plan_features import extract as planf
from src.utils.csv_manager import append_row
from src.utils.duplicate_checker import DuplicateTracker
from src.utils.logger import log
from pathlib import Path

Path("plans").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

Path("data/output").mkdir(parents=True, exist_ok=True)
Path("data/failed").mkdir(parents=True, exist_ok=True)

def run_pipeline():
 cfg=json.load(open('config/config.json'))
 conn=get_connection(cfg['database'])
 out='data/output/final_dataset.csv'
 machine=socket.gethostname()

 # Load duplicate tracker once into memory
 tracker=DuplicateTracker(out)

 # Find all batch files and sort them
 batch_files=sorted(glob.glob('data/input/Batch*.csv'))
 if not batch_files:
  print('No batch files found in data/input/')
  return

 print(f'Found {len(batch_files)} batch files')
 total_done=0
 total_skip=0
 total_fail=0

 for batch_file in batch_files:
  batch_name=os.path.basename(batch_file)
  qs=pd.read_csv(batch_file)
  print(f'\nProcessing {batch_name} ({len(qs)} queries)...')
  batch_done=0

  for _,q in qs.iterrows():
   for run in range(1,cfg['runs_per_query']+1):
    if tracker.exists(q['query_id'],run,machine):
     total_skip+=1
     continue
    try:
     cur=conn.cursor()
     cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) '+q['sql_text'])
     plan=cur.fetchone()[0][0]
     open(f"plans/{q['query_id']}_{run}.json",'w').write(json.dumps(plan))
     row={
       'machine_name':machine,'collected_by':cfg['collected_by'],
       'timestamp':datetime.now().isoformat(),
       'query_id':q['query_id'],'template_id':q['template_id'],
       'run_number':run,**sqlf(q['sql_text']),**sysf(),**planf(plan),
       'execution_time':plan['Plan'].get('Actual Total Time',0)
     }
     append_row(row,out)
     tracker.add(q['query_id'],run,machine)
     batch_done+=1
     total_done+=1
     log(f"DONE {q['query_id']} run {run}")
    except Exception as e:
     total_fail+=1
     with open('data/failed/failed_queries.csv','a') as f:
       f.write(f"{q['query_id']},{run},{batch_name},{e}\n")
     log(f"FAIL {q['query_id']} run {run}: {e}")
    time.sleep(cfg['sleep_between_runs'])

  print(f'  {batch_name}: {batch_done} new rows collected')

 conn.close()
 print(f'\nCompleted! Done={total_done}, Skipped={total_skip}, Failed={total_fail}')
