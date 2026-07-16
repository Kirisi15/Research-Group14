
import json,time,socket,pandas as pd
from datetime import datetime
from src.database.connection import get_connection
from src.features.semantic_features import extract as sqlf
from src.features.system_features import extract as sysf
from src.features.plan_features import extract as planf
from src.utils.csv_manager import append_row
from src.utils.duplicate_checker import exists
from src.utils.logger import log
from pathlib import Path

Path("plans").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

Path("data/output").mkdir(parents=True, exist_ok=True)
Path("data/failed").mkdir(parents=True, exist_ok=True)

def run_pipeline():
 cfg=json.load(open('config/config.json'))
 conn=get_connection(cfg['database'])
 qs=pd.read_csv('data/input/Batch01.csv')
 out='data/output/final_dataset.csv'
 machine=socket.gethostname()

 for _,q in qs.iterrows():
  for run in range(1,cfg['runs_per_query']+1):
   if exists(out,q['query_id'],run,machine): continue
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
    log(f"DONE {q['query_id']} run {run}")
   except Exception as e:
    with open('data/failed/failed_queries.csv','a') as f:
      f.write(f"{q['query_id']},{e}\n")
   time.sleep(cfg['sleep_between_runs'])
 conn.close()
 print('Completed')
