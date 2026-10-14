"""Build-only controlled candidate; never starts GPU or submits."""
import ast, hashlib, json
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
HERE=Path(__file__).resolve().parent
raw=(HERE/'candidate/r7.ipynb').read_bytes()
build=json.loads((HERE/'build.json').read_text())
assert hashlib.sha256(raw).hexdigest()==build['source_sha256']
manifest=json.loads((HERE/'assets/manifest.json').read_text())
for name in ('fluid_stacker.json','state_stacker.json'):
    assert hashlib.sha256((HERE/'assets'/name).read_bytes()).hexdigest()==manifest['sha256'][name]
fluid=json.loads((HERE/'assets/fluid_stacker.json').read_text())
state=json.loads((HERE/'assets/state_stacker.json').read_text())
assert fluid['classes']['Effusion']['adopt'] and not fluid['classes']['Effusion']['rule_adopt']
assert not any(v['adopt'] and v.get('joint_with_fluid') for v in state['classes'].values())
nb=json.loads(raw)
code="""# Controlled hypothesis: remove only the forced fluid-volume correction.
assert '--fluid-stacker' in WORKER_ARGS
_fluid_i=WORKER_ARGS.index('--fluid-stacker')
assert WORKER_ARGS[_fluid_i+1].endswith('fluid_stacker.json')
del WORKER_ARGS[_fluid_i:_fluid_i+2]
print('EXPERIMENT_NO_FLUID: R7 weights, Baker state correction, and 50/50 blend unchanged')
"""
nb['cells'].insert(6,nbformat.v4.new_code_cell(code))
for i,c in enumerate(nb['cells']):
    c['id']=f'r7-no-fluid-{i}'
    if c['cell_type']=='code':ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
nbformat.validate(nbformat.from_dict(nb))
run_dir=HERE/'no-fluid-run'
assert not (run_dir/'launch_attempt.json').exists()
out=run_dir/'candidate';out.mkdir(parents=True,exist_ok=True)
payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
(out/'r7.ipynb').write_bytes(payload)
meta=json.loads((HERE/'candidate/kernel-metadata.json').read_text())
meta.update(id='easoncyy/rsna-r7-no-fluid-control',title='RSNA R7 No Fluid Control')
(out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
record=dict(build)
record.update(status='BUILT_NOT_RUN',source_sha256=hashlib.sha256(payload).hexdigest(),parent_sha256=build['source_sha256'],hypothesis='Remove author-overridden Effusion correction; retain learned R7 and Baker correction',inference_unchanged=False)
(run_dir/'build.json').write_text(json.dumps(record,indent=2))
print('No-fluid controlled candidate BUILT_NOT_RUN')
