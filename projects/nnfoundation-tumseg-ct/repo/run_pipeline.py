"""Sequential frozen-protocol train, unseen inference, and independent evaluation."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys
from project_env import PROJECT

def main():
    logs=PROJECT/'experiments/pilot';logs.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',pipeline_pid=os.getpid(),started=dt.datetime.now().astimezone().isoformat(),stages=[])
    statepath=logs/'status.json'
    env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONUNBUFFERED']='1'
    for script,args in [('plan_data.py',['--profile','pilot']),('train.py',['--profile','pilot']),('predict.py',[]),('evaluate.py',[])]:
        cmd=[sys.executable,'-u',str(PROJECT/'repo'/script),*args]
        record=dict(stage=script,status='running',command=cmd,started=dt.datetime.now().astimezone().isoformat());state['stages'].append(record)
        with open(logs/(script.removesuffix('.py')+'.log'),'w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=p.pid;statepath.write_text(json.dumps(state,indent=2),encoding='utf-8')
            print('STAGE_STARTED',script,p.pid,flush=True);code=p.wait()
        record.update(returncode=code,status='completed' if code==0 else 'failed',finished=dt.datetime.now().astimezone().isoformat())
        if code:state['status']='failed'
        statepath.write_text(json.dumps(state,indent=2),encoding='utf-8')
        if code:raise SystemExit(code)
        print('STAGE_COMPLETE',script,flush=True)
    state.update(status='completed',finished=dt.datetime.now().astimezone().isoformat())
    statepath.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
