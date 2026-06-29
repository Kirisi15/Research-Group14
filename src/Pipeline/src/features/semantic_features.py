
import re
def extract(sql):
 s=sql.upper(); joins=len(re.findall(r'\bJOIN\b',s))
 return {
 'join_count':joins,'where_count':len(re.findall(r'\bWHERE\b',s)),
 'group_by_count':len(re.findall(r'GROUP\s+BY',s)),
 'order_by_count':len(re.findall(r'ORDER\s+BY',s)),
 'subquery_count':max(0,len(re.findall(r'\bSELECT\b',s))-1),
 'query_length':len(sql),
 'aggregate_count':len(re.findall(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(',s)),
 'distinct_count':len(re.findall(r'\bDISTINCT\b',s)),
 'condition_count':len(re.findall(r'\bAND\b|\bOR\b',s)),
 'table_count':len(re.findall(r'\bFROM\b',s))+joins}
