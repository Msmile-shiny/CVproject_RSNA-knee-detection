"""Read-only bounded community refresh; never launches notebooks."""
import ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
prepare_dns();a=KaggleApi();a.authenticate()
latest=[k.to_dict() for k in a.kernels_list(competition='rsna-knee-abnormality-detection',sort_by='dateRun',page_size=30)]
(HERE/'latest.json').write_text(json.dumps(latest,indent=2),encoding='utf8')
refs=['goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957','heliosli/blend-mine-in-private-lb-luck-to-all-upvoters','prvsiyan/the-bee-s-knees-final-rsna-push','ltcsmokeboi/rsna-knee-gp958e']
for ref in refs:
    out=HERE/ref.split('/')[0];out.mkdir(exist_ok=True)
    a.kernels_pull(ref,path=str(out),metadata=True)
    meta=json.loads((out/'kernel-metadata.json').read_text())
    source=out/meta['code_file']
    code=source.read_text(encoding='utf8')
    if source.suffix=='.ipynb':code='\n'.join(''.join(c['source']) for c in json.loads(code)['cells'])
    lines=[l for l in code.splitlines() if len(l)<450 and any(w in l.lower() for w in ['weight','0.95','blend','checkpoint','model','fallback','rank','r7','r6','mil','except'])]
    print(json.dumps(dict(ref=ref,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),datasets=meta.get('dataset_sources'),lines=lines[-130:]),ensure_ascii=False),flush=True)
