
def walk(n,s):
 if not isinstance(n,dict): return
 t=n.get('Node Type','')
 s['total_nodes']+=1
 if t=='Seq Scan': s['seq_scan_count']+=1
 if 'Index' in t: s['index_scan_count']+=1
 if t=='Hash Join': s['hash_join_count']+=1
 if t=='Nested Loop': s['nested_loop_count']+=1
 if t=='Merge Join': s['merge_join_count']+=1
 if t=='Sort': s['sort_count']+=1
 if 'Aggregate' in t: s['aggregate_node_count']+=1
 for c in n.get('Plans',[]): walk(c,s)

def extract(plan):
 p=plan['Plan']
 s={'seq_scan_count':0,'index_scan_count':0,'hash_join_count':0,
 'nested_loop_count':0,'merge_join_count':0,'sort_count':0,
 'aggregate_node_count':0,'total_nodes':0}
 walk(p,s)
 s.update({
 'plan_rows':p.get('Plan Rows',0),
 'actual_rows':p.get('Actual Rows',0),
 'node_type':p.get('Node Type',''),
 'parallel_aware':int(bool(p.get('Parallel Aware',False)))
 })
 return s
