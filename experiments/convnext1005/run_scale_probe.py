"""Launch or collect the recorded public inference diagnostic; never submit."""
import argparse
import datetime
import hashlib
import json
import socket
import subprocess
import ipaddress
import csv
import math
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

HERE=Path(__file__).resolve().parent
KERNEL='easoncyy/rsna-cnx-scale-probe'

def prepare_dns():
    """Process-local fallback for the API and observed output host; preserve TLS."""
    original=socket.getaddrinfo
    allowed={'api.kaggle.com','www.kaggleusercontent.com'}
    resolved={}
    def lookup(host,*args,**kwargs):
        if host in resolved: return original(resolved[host],*args,**kwargs)
        try: return original(host,*args,**kwargs)
        except socket.gaierror:
            if host not in allowed: raise
            command=f'Resolve-DnsName {host} -Server 1.1.1.1 -Type A -DnsOnly -QuickTimeout | Where-Object IPAddress | Select-Object -First 1 -ExpandProperty IPAddress'
            address=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],timeout=20,text=True).strip()
            ipaddress.IPv4Address(address);resolved[host]=address
            print(f'Using fresh process-local DNS for {host}; TLS verification unchanged',flush=True)
            return original(address,*args,**kwargs)
    socket.getaddrinfo=lookup

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','collect']);args=parser.parse_args()
    prepare_dns()
    api=KaggleApi();api.authenticate()
    raw=(HERE/'scale-probe/scale.ipynb').read_bytes()
    built=json.loads((HERE/'scale_build.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==built['source_sha256']
    if args.action=='launch':
        attempt=HERE/'scale_attempt.json';record=HERE/'scale_launch.json'
        assert not attempt.exists() and not record.exists(), 'Reconcile existing attempt; do not duplicate'
        assert not any(k.ref==KERNEL for k in api.kernels_list(mine=True,search='rsna-cnx-scale-probe',page_size=100))
        payload=dict(kernel=KERNEL,source_sha256=built['source_sha256'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        attempt.write_text(json.dumps(payload,indent=2))
        result=api.kernels_push(str(HERE/'scale-probe'),acc='NvidiaTeslaT4').to_dict()
        record.write_text(json.dumps(dict(**payload,response=result),indent=2))
        print(json.dumps(result))
    else:
        record=json.loads((HERE/'scale_launch.json').read_text())
        kernel=record['response']['ref'].removeprefix('/code/')
        state=api.kernels_status(kernel).to_dict();print(state,flush=True)
        if state['status'] not in ['COMPLETE','ERROR']: raise SystemExit(0)
        version=record['response']['versionNumber']
        out=HERE.parents[1]/'results'/f'cnx_scale_v{version}';out.mkdir(parents=True,exist_ok=True)
        api.kernels_output(kernel,path=str(out),file_pattern=r'^scale_(progress\.json|outcome\.json|receipt\.json|worker\.log|predictions\.csv)$',page_size=100)
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
        with api.build_kaggle_client() as client: remote=client.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.to_dict()['currentVersionNumber']==version
        codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
        result=dict(kernel=kernel,version=version,api_status=state['status'],source_verified=True,hidden_root_cause_confirmed=False)
        for name in ['scale_progress','scale_outcome','scale_receipt']:
            path=out/(name+'.json')
            if path.exists(): result[name]=json.loads(path.read_text())
        if state['status']=='COMPLETE':
            receipt=result['scale_receipt']
            assert receipt['study_count']==1024 and receipt['model_count']==3
            assert receipt['fallback']==0 and receipt['partial_predictions']==0
            assert result['scale_outcome']['status']=='COMPLETE'
            assert hashlib.sha256((out/'scale_predictions.csv').read_bytes()).hexdigest()==receipt['raw_sha256']
            assert all(receipt['source_hashes'][k]==v for k,v in built['model_source_hashes'].items())
            with (out/'scale_predictions.csv').open(newline='') as handle:
                reader=csv.DictReader(handle);rows=list(reader)
                assert reader.fieldnames[0]=='StudyInstanceUID' and len(reader.fieldnames)==13
            assert [r['StudyInstanceUID'] for r in rows]==built['studies']
            assert all(math.isfinite(float(v)) and 0<=float(v)<=1 for r in rows for k,v in r.items() if k!='StudyInstanceUID')
        (HERE/'scale_result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result))
