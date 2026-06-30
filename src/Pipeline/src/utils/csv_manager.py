
import pandas as pd, os
def append_row(row,path):
 df=pd.DataFrame([row])
 df.to_csv(path,mode='a',header=not os.path.exists(path),index=False)
