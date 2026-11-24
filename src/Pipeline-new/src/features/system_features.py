
import psutil
def extract():
 d=psutil.disk_io_counters()
 return {'cpu_percent':psutil.cpu_percent(),
 'memory_used_mb':round(psutil.virtual_memory().used/1048576,2),
 'disk_read_bytes':d.read_bytes,'disk_write_bytes':d.write_bytes}
