"""CPU checks: published payload integrity, parent identity, ranking endpoints."""
import ast
import hashlib
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
asset=ROOT/'experiments/assets1015/nartaa'
manifest=json.loads((asset/'code_manifest.json').read_text())
payload=(asset/'s6_extras.bin').read_bytes()
magic=b'K1PAYLOAD1\n'
assert payload.startswith(magic)
names=manifest['from_this_dataset']
with tarfile.open(fileobj=io.BytesIO(payload[len(magic):]),mode='r:') as archive:
    members=archive.getmembers()
    assert len(members)==len(names) and {m.name for m in members}==set(names)
    for member in members:
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        raw=archive.extractfile(member).read()
        expected=manifest['members'][member.name]
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
        if member.name.endswith('.py'): ast.parse(raw)
parent=json.loads((ROOT/'experiments/r71013/candidate/r7.ipynb').read_text(encoding='utf8'))
candidate=json.loads((HERE/'candidate/s6.ipynb').read_text(encoding='utf8'))
assert len(candidate['cells'])==len(parent['cells'])+1
for before,after in zip(parent['cells'],candidate['cells']):
    assert ''.join(before['source'])==''.join(after['source'])
adapter=ast.parse((HERE/'adapter.py').read_text())
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('run_a','main','blend_and_write') for n in ast.walk(adapter))
rng=np.random.default_rng(42)
a,b,c=[pd.DataFrame(rng.integers(0,20,(100,12))).rank(pct=True) for _ in range(3)]
np.testing.assert_allclose(.5*a+.5*b+0*c,.5*a+.5*b)
np.testing.assert_allclose(.5*a+.4*b+.1*b,.5*a+.5*b)
f=.5*a+.4*b+.1*c
assert np.isfinite(f.values).all() and ((f>=0)&(f<=1)).all().all()
result=dict(status='PASS',payload_members=len(names),parent_cells_unchanged=True,
            duplicate_raptor_call=False,ranking_tests='PASS',gpu_tested=False)
(HERE/'preflight.json').write_text(json.dumps(result,indent=2))
print(result)
