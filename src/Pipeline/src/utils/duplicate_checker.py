
import pandas as pd, os
def exists(path,q,r,m):
 if not os.path.exists(path): return False
 df=pd.read_csv(path,usecols=['query_id','run_number','machine_name'])
 return ((df['query_id']==q)&(df['run_number']==r)&(df['machine_name']==m)).any()
