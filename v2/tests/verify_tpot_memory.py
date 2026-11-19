"""Verify real worker limits and nanny recovery without exhausting the container."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from v2.adapters import tpot_resource_profile, tpot_dask_client
from v2.common import write_json


def allocate_and_hold(gib):
    allocation=np.ones(int(gib*1024**3/8),dtype=np.float64)
    time.sleep(20)
    return float(allocation[0])


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('/code/config.json').read_text());profile=tpot_resource_profile(config)
    before=Path('/sys/fs/cgroup/memory.events').read_text()
    with tpot_dask_client(profile,output) as client:
        original=client.run(os.getpid);assert len(set(original.values()))==2
        task=client.submit(allocate_and_hold,3.7,pure=False)
        deadline=time.monotonic()+45;changed=False
        while time.monotonic()<deadline:
            current=set(client.scheduler_info()['workers'])
            if current!=set(original):changed=True;break
            time.sleep(.05)
        task.cancel()
        assert changed,'Nanny did not restart a worker that exceeded its RSS threshold'
        client.wait_for_workers(2,timeout=45)
        current=client.run(os.getpid)
        assert len(set(current.values()))==2 and set(current.values())!=set(original.values())
        assert client.submit(sum,[1,2,3]).result(timeout=10)==6
    after=Path('/sys/fs/cgroup/memory.events').read_text()
    def values(text):return {line.split()[0]:int(line.split()[1]) for line in text.splitlines()}
    assert values(after)['oom_kill']==values(before)['oom_kill']==0
    write_json(output/'report.json',dict(status='passed',profile=profile,nanny_worker_restart_observed=True,
               cluster_recovered=True,container_oom_kills=0,initial_worker_pids=original,replacement_worker_pids=current,
               actual_container_memory_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
               container_memory_peak=Path('/sys/fs/cgroup/memory.peak').read_text().strip()))
    print('PASS: 2 workers, 4 GiB each, nanny restarted excess-memory worker, cluster recovered, container OOM kills=0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
