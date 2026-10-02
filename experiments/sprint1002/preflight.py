"""Read-only CPU validation of the built notebook and its mounted asset list."""
import hashlib
import json
from build_outer70 import ROOT, PARENT, PARENT_SHA, make_candidate


def check():
    raw = (PARENT/'sprint-fourway.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA, 'Parent changed'
    path = ROOT/'outer70/outer70.ipynb'
    receipt = json.loads((ROOT/'build_receipt.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['notebook_sha256'], 'Candidate artifact changed'
    assert json.loads(path.read_bytes()) == make_candidate(json.loads(raw)), 'Unexpected notebook edits'
    old = json.loads((PARENT/'kernel-metadata.json').read_text())
    new = json.loads((ROOT/'outer70/kernel-metadata.json').read_text())
    for key in old.keys() | new.keys():
        if key not in ('id', 'title', 'code_file'):
            assert old.get(key) == new.get(key), 'Runtime/dependency drift: '+key
    assert new['id'] == 'easoncyy/rsna-sprint-outer70'
    assert new['code_file'] == path.name
    print(json.dumps({
        'status':'LOCAL_PREFLIGHT_PASS', 'gpu_launched':False,
        'code_cells_compiled':25, 'parent_sha256':PARENT_SHA,
        'mounts':{k:new[k] for k in ('dataset_sources','kernel_sources','model_sources','competition_sources')},
        'online_inventory_file':str(ROOT/'remote_audit.json'),
        'limitations':['Read remote_audit.json for timestamped online listing evidence', 'No model forward pass', 'No candidate score'],
    }, indent=2))


if __name__ == '__main__':
    check()
