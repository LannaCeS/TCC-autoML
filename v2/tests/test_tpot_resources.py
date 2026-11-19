import json
from pathlib import Path
import unittest
from v2.adapters import tpot_resource_profile


class TPOTResourceTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parents[1]/'config.json').read_text())

    def test_two_workers_share_eight_gib_and_leave_eight_gib_headroom(self):
        c=self.config();p=tpot_resource_profile(c)
        self.assertEqual(p['workers'],2)
        self.assertEqual(p['worker_memory_bytes'],4*1024**3)
        self.assertEqual(p['workers']*p['worker_memory_bytes']+p['driver_memory_reserve_bytes'],16*1024**3)
        self.assertEqual(p['threads_per_worker'],1)
        self.assertLess(p['memory_pause'],p['memory_terminate'])
        self.assertLess(p['memory_terminate'],1)

    def test_unbounded_or_overcommitted_workers_are_rejected(self):
        for workers,memory in [(0,4),(8,4),(4,4),(2,None),(2,0)]:
            c=self.config();c['tpot_execution']={'workers':workers,'worker_memory_gib':memory}
            with self.assertRaises(ValueError):tpot_resource_profile(c)


if __name__=='__main__':unittest.main()
