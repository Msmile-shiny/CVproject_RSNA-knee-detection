import json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
prepare_dns();a=KaggleApi();a.authenticate()
records=[]
for ref in ['heliosli/rsna-knee-r6-frozen-fe055406-w1','rostigor/rsna-knee-r6-fe055406']:
    row={'ref':ref}
    try:
        row['listing']=a.dataset_list_files(ref,page_size=100).to_dict()
        out=HERE/ref.split('/')[0];out.mkdir(exist_ok=True)
        a.dataset_download_file(ref,'manifest.json',path=str(out))
        row['manifest_sha256']=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
        row['manifest']=json.loads((out/'manifest.json').read_text())
    except Exception as e:row['error']=str(e)[:300]
    records.append(row)
    print(json.dumps(row)[:6500],flush=True)
(HERE/'assets.json').write_text(json.dumps(records,indent=2))
