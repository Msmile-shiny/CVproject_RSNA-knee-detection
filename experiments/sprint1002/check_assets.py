"""Match downloaded public manifests to the live file inventory and pinned parent."""
import hashlib
import json
from build_outer70 import ROOT, PARENT


def check():
    remote=json.loads((ROOT/'remote_audit.json').read_text(encoding='utf-8'))
    listed={a['ref']:{f['name'] for f in a['files']} for a in remote['assets'] if a['status']=='LISTED'}
    source=(PARENT/'sprint-fourway.ipynb').read_text(encoding='utf-8')
    mapping={'d4':('mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2',''),
             'repair':('mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2','repairv1/'),
             'resgated':('mattiaangeli/rsna-knee-coat-resgated-ep10-top3',''),
             'global96':('mattiaangeli/rsna-knee-coatnet-global96-top3','')}
    findings=[]
    for kind,(ref,prefix) in mapping.items():
        paths=list((ROOT/'asset_manifests'/kind).rglob('*.json'))
        assert len(paths)==1,kind
        p=paths[0];m=json.loads(p.read_text(encoding='utf-8'))
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha in source, 'Manifest not pinned in parent: '+kind
        missing=[prefix+name for name in m['files'] if prefix+name not in listed[ref]]
        assert not missing,(kind,missing)
        findings.append(dict(family=kind,manifest_sha256=sha,pinned_in_parent=True,listed_files=len(m['files'])))
    dino=json.loads((ROOT/'asset_manifests/dino/manifest.json').read_text(encoding='utf-8'))
    assert len(dino['members'])==20
    assert all(m['file'] in listed['pilkwang/rsna-knee-weights'] for m in dino['members'])
    assert all(f'm_f{i}.pt' in listed['mattiaangeli/knee-mri-fold-weights'] for i in range(5))
    for ref,name in [('dreaddevelopment/raptor-knee-maxspan','raptor_ft_coatnet_v5_full_swa.pt'),
                     ('dreaddevelopment/raptor-knee-native384','raptor_ft_coatnet_v8_full_swa.pt'),
                     ('dreaddevelopment/raptor-knee-native384dense','raptor_ft_coatnet_v10_full.pt')]:
        assert name in listed[ref],name
    out=dict(status='MANIFEST_AND_NAMES_PASS',checked_at_utc=remote['checked_at_utc'],families=findings,
             dino_members=20,a5_folds=5,raptor_checkpoints=3,
             dino_pixel_groups=sorted({m['pixel_group'] for m in dino['members']}),
             checkpoint_bytes_rehashed_locally=False,
             note='Manifest bytes checked; checkpoint content checks remain in the mandatory runtime loaders.')
    (ROOT/'asset_check_receipt.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':check()
