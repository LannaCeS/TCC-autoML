"""Full-volume validation-only checks using immutable compatible preparations."""
import fcntl
import json
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2.orchestrate import ROOT,freeze_inputs,docker,slot_context
from v2.protocol import validate_config


def main():
    config=validate_config(json.loads((ROOT/'config.json').read_text()))
    lock=(ROOT/'outputs/.benchmark.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=Path(json.loads((ROOT/'outputs/active-official.json').read_text())['output_directory'])
    folder=ROOT/'outputs'/('tpot-memory-qualification-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    (folder/'runs').mkdir(parents=True)
    context=freeze_inputs(folder,['tpot'],config)
    scenarios=['random_over','smote','smotetomek']
    def run(index,scenario):
        output=folder/'runs'/f'tpot-{scenario}-101';output.mkdir()
        prepared=previous/'prepared'/f'{scenario}-101'
        command=['/code/run.py','--framework','tpot','--scenario',scenario,'--seed','101',
                 '--prepared','/outputs/'+str(prepared.relative_to(ROOT/'outputs')),
                 '--output','/outputs/'+str(output.relative_to(ROOT/'outputs')),
                 '--budget','300','--validation-only']
        code=docker('tpot',command,output/'execution.log',context=slot_context(context,(index,config['execution_slots'][index])),timeout=1320)
        state=json.loads((output/'status.json').read_text());print(scenario,state['status'],flush=True)
        assert code==0 and state['status']=='qualified',(scenario,state)
        model=json.loads((output/'selection.json').read_text())
        assert model['execution_resources']['workers']==2 and model['execution_resources']['worker_memory_bytes']==4*1024**3
        workers=json.loads((output/'tpot-worker-configuration.json').read_text())['workers']
        assert len(workers)==2 and all(w['memory_limit']==4*1024**3 for w in workers.values())
        events=json.loads((output/'tpot-memory-events.json').read_text())['memory_events']
        assert int(dict(line.split() for line in events.splitlines())['oom_kill'])==0
        assert 'test' not in json.loads((output/'metrics.json').read_text())
        return dict(scenario=scenario,status='qualified',run=str(output.relative_to(ROOT)),container_oom_kills=0,
                    search_wall_seconds=state['search_wall_seconds'],memory_peak_bytes=int(json.loads((output/'resources.json').read_text())['memory.peak']))
    print('Output:',folder,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks=list(pool.map(lambda pair:run(*pair),enumerate(scenarios)))
    (folder/'report.json').write_text(json.dumps(dict(status='passed',checks=checks,external_test_used=False),indent=2)+'\n')
    (ROOT/'outputs/tpot-memory-qualification-latest.json').write_text(json.dumps(dict(folder=str(folder),report=str(folder/'report.json')),indent=2)+'\n')
    print('PASS: all three full-data scenarios qualified without container OOM',flush=True)


if __name__=='__main__':main()
