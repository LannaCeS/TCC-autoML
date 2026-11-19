"""Bounded dependency scheduler and CPU isolation checks (standard library only)."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import threading


def cpu_ids(specification):
    result = []
    for part in specification.split(','):
        bounds = part.split('-')
        if len(bounds) == 1:
            result.append(int(bounds[0]))
        elif len(bounds) == 2:
            first, last = map(int, bounds)
            if first > last:
                raise ValueError('Invalid descending CPU range')
            result.extend(range(first, last + 1))
        else:
            raise ValueError('Invalid CPU specification')
    if len(set(result)) != len(result) or any(cpu < 0 for cpu in result):
        raise ValueError('CPU IDs must be distinct and nonnegative')
    return set(result)


def validate_slots(specifications, cpus=8, topology=None, allowed=None):
    sets = [cpu_ids(spec) for spec in specifications]
    if not sets or any(len(ids) != cpus for ids in sets):
        raise ValueError('Every execution slot must contain exactly eight logical CPUs')
    seen, physical = set(), set()
    for ids in sets:
        if seen & ids:
            raise ValueError('Execution slots overlap in logical CPUs')
        if allowed is not None and not ids <= set(allowed):
            raise ValueError('Execution slot contains unavailable CPUs')
        seen.update(ids)
        if topology is not None:
            cores = {topology[cpu] for cpu in ids}
            if physical & cores:
                raise ValueError('Execution slots share a physical core through SMT siblings')
            physical.update(cores)
    return sets


def host_slots(specifications, cpus, memory_gib):
    topology = {}
    for cpu in set.union(*(cpu_ids(s) for s in specifications)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[cpu] = ((base / 'physical_package_id').read_text().strip(),
                         (base / 'core_id').read_text().strip())
    allowed = os.sched_getaffinity(0)
    validate_slots(specifications, cpus, topology, allowed)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    limit_path = Path('/sys/fs/cgroup/memory.max')
    current_path = Path('/sys/fs/cgroup/memory.current')
    if limit_path.exists() and limit_path.read_text().strip() != 'max':
        available = min(available, int(limit_path.read_text()) - int(current_path.read_text()))
    required = (len(specifications) * memory_gib + 8) * 1024**3
    if available < required:
        raise ValueError('Insufficient available memory for all container limits plus an 8 GiB host reserve')
    cpu_limit = Path('/sys/fs/cgroup/cpu.max')
    if cpu_limit.exists():
        quota, period = cpu_limit.read_text().split()
        if quota != 'max' and int(quota) / int(period) < len(specifications) * cpus:
            raise ValueError('Host cgroup CPU quota is below the requested parallel allocation')
    return {'slots': specifications, 'logical_cpus_per_slot': cpus,
            'memory_gib_per_slot': memory_gib, 'memory_available_bytes_at_start': available,
            'physical_cores': {str(cpu): list(core) for cpu, core in sorted(topology.items())},
            'no_logical_or_physical_core_overlap': True}


def schedule(groups, slots, prepare, experiment, progress, stop=None, priority=None):
    """Preparation and dependent experiments share the same bounded slot pool."""
    stop = stop if stop is not None else threading.Event()
    ready = deque(('prepare', group, None) for group in groups)
    available = deque(enumerate(slots))
    running, failures = {}, 0
    pool = ThreadPoolExecutor(max_workers=len(slots))
    try:
        while (ready or running) and not stop.is_set():
            if priority is not None:
                ready = deque(sorted(ready, key=priority))
            while ready and available and not stop.is_set():
                task = ready.popleft()
                slot = available.popleft()
                kind, group, tool = task
                future = pool.submit(prepare if kind == 'prepare' else experiment,
                                     group, slot, *([] if kind == 'prepare' else [tool]))
                running[future] = (task, slot)
            done, _ = wait(running, timeout=.5, return_when=FIRST_COMPLETED)
            for future in done:
                (kind, group, tool), slot = running.pop(future)
                code = future.result()
                failures += bool(code)
                available.append(slot)
                if kind == 'prepare' and not code:
                    ready.extendleft(reversed([('experiment', group, tool) for tool in group['frameworks']]))
                progress()
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return failures
