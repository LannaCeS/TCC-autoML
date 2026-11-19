import threading
import time
import unittest
from v2.parallel import cpu_ids, validate_slots, schedule


class ParallelTests(unittest.TestCase):
    def test_slots_reject_overlap_including_smt_siblings(self):
        self.assertEqual(cpu_ids('0-3,8-11'), {0, 1, 2, 3, 8, 9, 10, 11})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_slots(['0-7', '4-11'])
        topology = {i: i // 2 for i in range(16)}
        with self.assertRaisesRegex(ValueError, 'physical core'):
            validate_slots(['0,2,4,6,8,10,12,14', '1,3,5,7,9,11,13,15'], topology=topology)
        validate_slots(['0-7', '8-15'], topology=topology, allowed=set(range(16)))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            validate_slots(['0-7'], allowed=set(range(4)))

    def test_dependencies_and_maximum_concurrency_and_slot_reuse(self):
        lock, barrier = threading.Lock(), threading.Barrier(2)
        active, maximum, prepared, completed, running_slots = 0, 0, set(), [], set()
        def enter(slot):
            nonlocal active, maximum
            with lock:
                self.assertNotIn(slot[0], running_slots)
                running_slots.add(slot[0]); active += 1; maximum = max(maximum, active)
        def leave(slot):
            nonlocal active
            with lock:
                running_slots.remove(slot[0]); active -= 1
        def prepare(group, slot):
            enter(slot); barrier.wait(timeout=3)
            with lock: prepared.add(group['name'])
            leave(slot); return 0
        def run(group, slot, tool):
            enter(slot)
            with lock: self.assertIn(group['name'], prepared)
            time.sleep(.01)
            with lock: completed.append((group['name'], tool))
            leave(slot); return 0
        groups = [dict(name=name, frameworks=['a', 'b', 'c']) for name in ['one', 'two']]
        self.assertEqual(schedule(groups, ['0-7', '8-15'], prepare, run, lambda: None), 0)
        self.assertEqual(maximum, 2)
        self.assertEqual(len(completed), 6)
        self.assertEqual(active, 0)

    def test_retry_priority_is_applied_before_ordinary_ready_groups(self):
        seen=[]
        groups=[dict(name='normal',frameworks=['a']),dict(name='retry',frameworks=['b'])]
        schedule(groups,['0-7'],lambda group,slot:seen.append('prepare:'+group['name']) or 0,
                 lambda group,slot,tool:seen.append('run:'+tool) or 0,lambda:None,
                 priority=lambda task:0 if task[1]['name']=='retry' else 1)
        self.assertEqual(seen[:2],['prepare:retry','run:b'])

    def test_failed_preparation_never_launches_its_experiments(self):
        ran = []
        groups = [dict(name='bad', frameworks=['a']), dict(name='good', frameworks=['b'])]
        failures = schedule(groups, ['0-7'], lambda group, slot: int(group['name'] == 'bad'),
                            lambda group, slot, tool: ran.append(tool) or 0, lambda: None)
        self.assertEqual(failures, 1)
        self.assertEqual(ran, ['b'])

    def test_cancel_stops_pending_tasks_and_reports_worker_exception(self):
        stop = threading.Event()
        def prepare(group, slot):
            raise RuntimeError('worker failure')
        with self.assertRaisesRegex(RuntimeError, 'worker failure'):
            schedule([dict(frameworks=['unused'])], ['0-7'], prepare,
                     lambda *args: self.fail('dependent task launched'), lambda: None, stop)
        self.assertTrue(stop.is_set())


if __name__ == '__main__':
    unittest.main()
