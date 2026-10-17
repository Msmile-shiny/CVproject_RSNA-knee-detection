import json
from pathlib import Path
for folder in ['new','s6review']:
    p=Path(__file__).parent/folder
    m=json.loads((p/'kernel-metadata.json').read_text())
    n=json.loads((p/m['code_file']).read_text())
    print(folder,m)
    for i,c in enumerate(n['cells']):
        s=''.join(c['source'])
        if c['cell_type']=='markdown':print(s[:9000])
        else:
            print('CELL',i)
            print('\n'.join(l for l in s.splitlines() if len(l)<450 and any(x in l for x in ['W_RAPTOR_C =','A224 =','C_SLUG =','FLIP_TTA =','S6','OWN_W','0.958','0.957','R7W'])))
