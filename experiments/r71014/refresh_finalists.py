"""Read-only remote reconciliation of scored finalists. Never selects or submits."""
import hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CASES=[
 ('r7_original',57144024,357976215,'r71013','candidate/r7.ipynb'),
 ('r7_no_fluid',57147149,357997977,'r71013/no-fluid-run','candidate/r7.ipynb'),
 ('r7_mirror',57185504,358219734,'r71013/flip-run','candidate/r7.ipynb'),
 ('r7_threeway',57197910,358368480,'r71014','candidate/threeway.ipynb'),
 ('r6_fallback',57131878,357879252,'community1012','candidate/compact.ipynb'),
]

def classify(row):
    if row.get('errorDescription'):return 'FAILED',None
    if row.get('status')!='COMPLETE':return 'AWAITING',None
    try:score=float(row.get('publicScore'))
    except (ValueError,TypeError):return 'AWAITING_SCORE',None
    if not math.isfinite(score) or not 0<=score<=1:return 'INVALID_SCORE',None
    return 'SCORED',score

def main():
    sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    prepare_dns();a=KaggleApi();a.authenticate()
    rows={int(s.to_dict()['ref']):s.to_dict() for s in a.competition_submissions('rsna-knee-abnormality-detection')}
    records=[]
    for name,ref,script,folder,source in CASES:
        p=ROOT/'experiments'/folder
        receipt=json.loads((p/'submission.json').read_text())
        assert receipt['response']['ref']==ref
        built=json.loads((p/'build.json').read_text())
        h=hashlib.sha256((p/source).read_bytes()).hexdigest()
        assert h==built['source_sha256'],name+' source drift'
        row=rows.get(ref)
        assert row is not None,'Missing submission; do not overwrite prior results'
        assert row['url']==f"/code/{receipt['kernel']}?scriptVersionId={script}"
        state,score=classify(row)
        records.append(dict(name=name,submission_ref=ref,scriptVersionId=script,kernel=receipt['kernel'],
                            source_sha256=h,state=state,publicScore=score,raw=row))
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),candidates=records,
                protected_original=57144024,final_selection_changed=False,
                note='Displayed scores cannot establish exact ties or Private performance. No automatic final selection.')
    (ROOT/'experiments/r71014/finalists.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
