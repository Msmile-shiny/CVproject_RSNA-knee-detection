"""Verify a completed exact Notebook version, collect small outputs, submit once."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
PARENT_SHA='d0165cf3d144b6be4e05ece77f80b233c27393cdbae67cb06c0d03b96f178505'
CONTRACTS={
    'meniscus':dict(folder='sprint1005',notebook='meniscus10',receipt='meniscus10_receipt.json',files=['submission.csv','parent943_replay.csv','meniscus10_receipt.json','sprint_fourway_receipt.json','btkd_v559_complete.json','meniscus10/public0033_bag_raw.csv','meniscus10/public0033_cached_inference_receipt.json','meniscus10/specialist_receipt.json']),
    'cnx3':dict(folder='convnext1005',notebook='cnx3fold30',receipt='cnx3_receipt.json',files=['submission.csv','parent943_replay.csv','cnx3_receipt.json','sprint_fourway_receipt.json','btkd_v559_complete.json','cnx3_raw.csv','cnx3_worker_receipt.json']),
}

def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def check(kind,out):
    contract=CONTRACTS[kind]
    build=read_json(ROOT/'experiments'/contract['folder']/'build_receipt.json')
    receipt=read_json(out/contract['receipt'])
    if receipt.get('status')!='COMPLETE' or receipt.get('ready_for_scoring') is not True or receipt.get('parent_sha256')!=PARENT_SHA:
        raise ValueError('Candidate incomplete or parent identity changed')
    elapsed=float(receipt.get('elapsed_seconds',float('nan')))
    if not np.isfinite(elapsed) or not 0<elapsed<8.5*3600: raise ValueError('Invalid total runtime')
    for name,digest in receipt['hashes'].items():
        if name not in contract['files'] or file_sha(out/name)!=digest: raise ValueError('Output identity mismatch: '+name)
    parent_receipt=read_json(out/'sprint_fourway_receipt.json')
    if parent_receipt.get('ready_for_scoring') is not True or set(parent_receipt.get('family_members',[]))!={'resgated_top3','global96_top3','d4_swa3','repairv1_top3'}:
        raise ValueError('Parent model set incomplete')
    audit=read_json(out/'btkd_v559_complete.json')
    flags=('fail','fallback','unavailable','reject','incomplete','partial','mismatch','dropped','neutral')
    issues=[e for e in audit.get('events',[]) if any(f in e.get('kind','').lower() for f in flags)]
    if issues: raise ValueError('Parent diagnostic failures: '+str(issues)[:500])
    if file_sha(out/'parent943_replay.csv')!=parent_receipt.get('submission_sha256'):
        raise ValueError('Parent replay differs from the verified parent output')
    parent=pd.read_csv(out/'parent943_replay.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
    final=pd.read_csv(out/'submission.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
    for frame in (parent,final):
        if frame.columns.tolist()!=['StudyInstanceUID',*LABELS] or frame.empty or frame.StudyInstanceUID.isna().any() or frame.StudyInstanceUID.duplicated().any(): raise ValueError('Bad frame schema/UID')
        a=frame[LABELS].to_numpy()
        if not np.isfinite(a).all() or (a<0).any() or (a>1).any(): raise ValueError('Invalid output values')
    if parent.StudyInstanceUID.tolist()!=final.StudyInstanceUID.tolist() or len(final)!=receipt['study_count']: raise ValueError('Study order/count differs')
    if kind=='meniscus':
        if receipt['specialist_script_sha256']!=build['specialist_script_sha256'] or receipt['overlay_sha256']!=build['overlay_sha256'] or receipt['specialist_weight']!=.1: raise ValueError('Specialist recipe drift')
        worker=read_json(out/'meniscus10/specialist_receipt.json')
        raw=read_json(out/'meniscus10/public0033_cached_inference_receipt.json')
        if worker.get('status')!='COMPLETE' or not worker.get('full_batch_stress_pass') or worker.get('stress_batch_studies')!=32: raise ValueError('Full batch stress missing')
        if worker.get('decode_failures')!=0 or worker.get('partial_predictions')!=0 or worker.get('source_sha256')!=build['author_source_sha256']: raise ValueError('Specialist preprocessing failures/source drift')
        if raw.get('status')!='passed' or raw.get('fallback')!=0 or raw.get('neutralized_predictions')!=0 or raw.get('study_count')!=len(final) or raw.get('precision')!='fp32' or raw.get('window_starts')!=list(range(10)) or not raw.get('checkpoint',{}).get('strict_load'): raise ValueError('Incomplete specialist runtime')
        if raw['checkpoint'].get('sha256')!='e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7' or raw['checkpoint'].get('state_tensor_count')!=233: raise ValueError('Specialist weight identity changed')
        bag=pd.read_csv(out/'meniscus10/public0033_bag_raw.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
        targets=['Medial Meniscus','Lateral Meniscus']
        if bag.columns.tolist()!=['StudyInstanceUID',*targets] or bag.StudyInstanceUID.duplicated().any() or set(bag.StudyInstanceUID)!=set(parent.StudyInstanceUID): raise ValueError('Specialist frame coverage')
        bag=bag.set_index('StudyInstanceUID').loc[parent.StudyInstanceUID].reset_index()
        expected=parent[LABELS].copy()
        expected[targets]=.9*parent[targets].rank(method='average',pct=True)+.1*bag[targets].rank(method='average',pct=True)
        import csv
        before=list(csv.reader((out/'parent943_replay.csv').open(newline='',encoding='utf-8')))
        after=list(csv.reader((out/'submission.csv').open(newline='',encoding='utf-8')))
        for i,label in enumerate(['StudyInstanceUID',*LABELS]):
            if label not in targets and any(a[i]!=b[i] for a,b in zip(before,after)): raise ValueError('Untouched token drift: '+label)
    else:
        worker=read_json(out/'cnx3_worker_receipt.json')
        if receipt.get('reader_weight')!=.3 or receipt.get('upstream_sha256')!=build['upstream_sha256'] or worker.get('status')!='COMPLETE' or worker.get('model_count')!=3 or worker.get('folds')!=[0,1,2] or not worker.get('stress_pass') or worker.get('fallback')!=0 or worker.get('partial_predictions')!=0: raise ValueError('ConvNeXt recipe/coverage failure')
        if worker.get('study_count')!=len(final) or worker.get('precision')!='fp16' or worker.get('batch_studies')!=4 or worker.get('windows_per_slot')!=12: raise ValueError('ConvNeXt execution contract changed')
        if any(worker['source_hashes'].get(k)!=v for k,v in build['sources'].items()): raise ValueError('ConvNeXt source differs')
        if set(worker.get('checkpoint_hashes',{}))!={f'cnxt_v0_fold{i}.pt' for i in range(3)} or len(set(worker['checkpoint_hashes'].values()))!=3: raise ValueError('ConvNeXt checkpoint set incomplete or duplicated')
        bag=pd.read_csv(out/'cnx3_raw.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
        if bag.columns.tolist()!=['StudyInstanceUID',*LABELS] or bag.StudyInstanceUID.duplicated().any() or set(bag.StudyInstanceUID)!=set(parent.StudyInstanceUID): raise ValueError('ConvNeXt frame coverage')
        bag=bag.set_index('StudyInstanceUID').loc[parent.StudyInstanceUID].reset_index()
        expected=(.7*parent[LABELS].rank(method='average',pct=True)+.3*bag[LABELS].rank(method='average',pct=True)).rank(method='average',pct=True)
    if not np.isfinite(bag.drop(columns='StudyInstanceUID').to_numpy()).all(): raise ValueError('Nonfinite reader')
    if not np.allclose(expected,final[LABELS],rtol=0,atol=1e-12): raise ValueError('Final blend differs from fixed recipe')
    return dict(status='VISIBLE_OUTPUT_CHECK_PASS',kind=kind,studies=len(final),elapsed_seconds=elapsed,hidden_runtime_verified=False,score_verified=False)

def finish(kind,submit=False):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    import re
    c=CONTRACTS[kind];folder=ROOT/'experiments'/c['folder']
    launch=read_json(folder/'launch_receipt.json')
    kernel=launch['response']['ref'].removeprefix('/code/')
    version=launch['response']['versionNumber']
    api=KaggleApi();api.authenticate()
    status=api.kernels_status(kernel).to_dict()
    print(kernel,status,flush=True)
    if status['status'] not in ('COMPLETE','ERROR'): return
    out=ROOT/'results'/f'sprint1005_{kind}_v{version}';out.mkdir(parents=True,exist_ok=True)
    pattern='^('+'|'.join(re.escape(f) for f in c['files'])+')$'
    api.kernels_output(kernel,path=str(out),file_pattern=pattern,page_size=500)
    if status['status']!='COMPLETE': raise RuntimeError('Kernel failed; inspect downloaded log, do not submit')
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with api.build_kaggle_client() as client: remote=client.kernels.kernels_api_client.get_kernel(req)
    metadata=remote.metadata.to_dict()
    if metadata.get('currentVersionNumber')!=version: raise RuntimeError('Latest version differs from recorded launch')
    meta=read_json(folder/c['notebook']/'kernel-metadata.json')
    raw=(folder/c['notebook']/meta['code_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=launch['notebook_sha256']: raise RuntimeError('Local launched source mutated')
    local=json.loads(raw);pulled=json.loads(remote.blob.source)
    codes=lambda n:[''.join(cell['source']) for cell in n['cells'] if cell['cell_type']=='code']
    if codes(local)!=codes(pulled): raise RuntimeError('Remote version code differs')
    (out/'verified_source.ipynb').write_text(remote.blob.source,encoding='utf-8')
    (out/'verified_metadata.json').write_text(json.dumps(metadata,indent=2))
    verified=check(kind,out)
    (folder/'visible_check_receipt.json').write_text(json.dumps(verified,indent=2))
    print(verified,flush=True)
    if submit:
        record=folder/'submission_receipt.json';attempt=folder/'submission_attempt.json'
        if record.exists(): print('Already submitted',record.read_text());return
        if attempt.exists(): raise RuntimeError('Previous attempt requires reconciliation')
        attempt.write_text(json.dumps(dict(kernel=kernel,version=version,check=verified),indent=2))
        result=api.competition_submit_code(file_name='submission.csv',message=f'Sprint1005 {kind}: fixed public reader addition to verified 0.943 parent; full coverage and source gates passed',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
        record.write_text(json.dumps(dict(kernel=kernel,version=version,response=result,check=verified),indent=2))
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=CONTRACTS);p.add_argument('--submit',action='store_true');args=p.parse_args()
    finish(args.kind,args.submit)
