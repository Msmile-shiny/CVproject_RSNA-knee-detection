"""Bounded read-only discovery of genuinely new public model assets."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
prepare_dns();a=KaggleApi();a.authenticate()
targets={
 'nartaa/rsna-knee-s6fall-weights':['README.md','checkpoint-manifest.json','E_prior_s42_fall.run_config.json'],
 'dkim54/rsna-knee-models-v11c':['models.json','code/model.py','code/dataset.py','code/infer_lib.py'],
}
records=[]
for ref,names in targets.items():
    out=HERE/ref.split('/')[0];out.mkdir(parents=True,exist_ok=True)
    listing=a.dataset_list_files(ref,page_size=100).to_dict()
    for name in names:a.dataset_download_file(ref,name,path=str(out))
    records.append(dict(ref=ref,listing=listing,local_sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}))
a.kernels_pull('nartaa/rsna-knee-s6fall-solo',path=str(HERE/'nartaa-notebook'),metadata=True)
record=dict(assets=records,dkim_kernels=[k.to_dict() for k in a.kernels_list(user='dkim54',search='knee',page_size=20)],gpu_started=False)
(HERE/'audit.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record['dkim_kernels']))
