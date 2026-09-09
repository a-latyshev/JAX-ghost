"""Optional phase snapshots and an external 100 ms memory sampler (stdlib only)."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


def proc_memory(pid):
    try:
        entries = {}
        for line in Path('/proc/{}/status'.format(pid)).read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM'):
                entries[key] = int(value.split()[0]) * 1024
        return entries.get('VmRSS'), entries.get('VmHWM')
    except (OSError, ValueError):
        return None, None


def mark(phase, device=None, details=None):
    directory = os.environ.get('BENCH_MEMORY_DIR')
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    rss, hwm = proc_memory(os.getpid())
    stats = None
    if device is not None:
        try:
            stats = device.memory_stats()
        except (AttributeError, RuntimeError, NotImplementedError):
            pass
    row = dict(timestamp=time.time(), pid=os.getpid(), rank=rank, phase=phase,
               rss_bytes=rss, lifetime_host_peak_bytes=hwm, allocator_stats=stats,
               details=details)
    path = root/('rank-{}.json'.format(rank))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row))
    temporary.replace(path)
    with (root/('rank-{}.jsonl'.format(rank))).open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()


def query_gpu_processes():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
            '--format=csv,noheader,nounits'], universal_newlines=True, timeout=3)
        rows = {}
        for row in csv.reader(result.splitlines()):
            if len(row) == 3:
                try:
                    rows[int(row[0])] = (row[1].strip(), int(row[2].strip())*1024**2)
                except ValueError:
                    continue
        return rows
    except (OSError, subprocess.SubprocessError):
        return {}


def monitor(root, cpu):
    os.sched_setaffinity(0, {cpu})
    fields = ['timestamp','rank','pid','phase','rss_bytes','lifetime_host_peak_bytes','gpu_uuid','gpu_process_bytes']
    with (root/'samples.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not (root/'stop').exists():
            began = time.monotonic()
            gpu = query_gpu_processes()
            timestamp = time.time()
            for path in sorted(root.glob('rank-*.json')):
                try:
                    state = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if state['phase'] in ('workload_done','diagnostics','done'):
                    continue
                rss, hwm = proc_memory(state['pid'])
                uuid, used = gpu.get(state['pid'], (None, None))
                writer.writerow(dict(timestamp=timestamp, rank=state['rank'], pid=state['pid'],
                    phase=state['phase'], rss_bytes=rss, lifetime_host_peak_bytes=hwm,
                    gpu_uuid=uuid, gpu_process_bytes=used))
            stream.flush()
            time.sleep(max(0, .1-(time.monotonic()-began)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    monitor(args.directory, args.cpu)
