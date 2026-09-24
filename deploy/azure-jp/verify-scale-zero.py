"""Verify automatic GPU shutdown while the CPU website remains available."""
import json
from pathlib import Path
import subprocess
import sys
import time
import httpx

AZ='az.cmd' if sys.platform=='win32' else 'az'
ROOT=Path(__file__).resolve().parents[2]
HOST='https://vidyi.cc'


def revisions():
    result=subprocess.run([AZ,'containerapp','revision','list','-g','videotranslator-jpe-rg',
        '-n','videotranslator-gpu','-o','json','--only-show-errors'],capture_output=True,text=True,check=True,timeout=60)
    return json.loads(result.stdout)


def main():
    started=time.time()
    last=None
    while time.time()-started<480:
        current=revisions()
        summary=[{'name':r['name'],'active':r['properties'].get('active'),
            'replicas':r['properties'].get('replicas',0),'state':r['properties'].get('runningState')} for r in current]
        if summary!=last:
            print(json.dumps(summary),flush=True)
            last=summary
        if current and all(r['properties'].get('replicas',0)==0 for r in current):
            response=httpx.get(HOST+'/api/v1/health/ready',timeout=30)
            assert response.status_code==200 and response.json()['ok']
            assert httpx.get(HOST,timeout=30).status_code==200
            report={'automatic_scale_zero':True,'website_ready_at_zero':True,'verified_at':time.time(),'revisions':summary}
            (ROOT/'vt-data/azure-tmp/gpu-scale-zero.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('GPU automatically returned to zero; HTTPS website remains healthy.',flush=True)
            return
        time.sleep(20)
    # A failed scaling test must not keep billing while awaiting investigation.
    for r in revisions():
        if r['properties'].get('active'):
            subprocess.run([AZ,'containerapp','revision','deactivate','-g','videotranslator-jpe-rg',
                '-n','videotranslator-gpu','--revision',r['name'],'--only-show-errors','-o','none'],check=True)
    raise TimeoutError('Automatic shutdown failed; revisions were manually deactivated')


if __name__=='__main__':
    main()
