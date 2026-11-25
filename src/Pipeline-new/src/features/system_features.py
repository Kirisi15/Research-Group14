import psutil


def extract():
    # Measure CPU utilization over a 1-second interval.
    # This function must be called before executing the SQL query.
    cpu_percent = psutil.cpu_percent(interval=1)

    # Capture system memory usage
    memory_used_mb = round(
        psutil.virtual_memory().used / 1048576, 2
    )

    # Capture cumulative disk I/O counters
    disk = psutil.disk_io_counters()

    return {
        "cpu_percent": cpu_percent,
        "memory_used_mb": memory_used_mb,
        "disk_read_bytes": disk.read_bytes,
        "disk_write_bytes": disk.write_bytes
    }
