"""Embed the exact preprocessing in a bounded CPU diagnostic notebook."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    sources={name:(HERE/name).read_text(encoding='utf-8') for name in ['preprocess.py','audit_public_data.py']}
    for source in sources.values(): ast.parse(source)
    assert hashlib.sha256(sources['preprocess.py'].encode()).hexdigest()=='4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2'
    code='''from pathlib import Path
import os, subprocess, sys
src=Path('/kaggle/working/public_audit_source');src.mkdir(exist_ok=True)
for name,source in SOURCES.items(): (src/name).write_text(source,encoding='utf-8')
roots={p.parent for pattern in ['datasets/*/*/cnxt_v0_fold0.pt','*/cnxt_v0_fold0.pt'] for p in Path('/kaggle/input').glob(pattern)}
assert len(roots)==1
envdir=Path('/kaggle/working/public_audit_env')
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--target',str(envdir),'--find-links',str(next(iter(roots))),'pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'],check=True)
env=dict(os.environ,PYTHONPATH=str(envdir),CUDA_VISIBLE_DEVICES='')
subprocess.run([sys.executable,'-u',str(src/'audit_public_data.py')],env=env,check=True,timeout=6*3600)
'''.replace('SOURCES',repr(sources))
    ast.parse(code)
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12.0'}})
    nbformat.validate(notebook)
    folder=HERE/'public-audit';folder.mkdir(exist_ok=True)
    nbformat.write(notebook,folder/'audit.ipynb')
    parent=json.loads((HERE/'cnx3fold30/kernel-metadata.json').read_text())
    metadata=dict(id='easoncyy/rsna-cnx-public-data-audit',title='RSNA CNX Public Data Audit',
                  code_file='audit.ipynb',language='python',kernel_type='notebook',is_private=True,
                  enable_gpu=False,enable_tpu=False,enable_internet=False,
                  dataset_sources=['goodpjw2008/rsna-knee-2-5d-convnext-reader'],
                  competition_sources=['rsna-knee-abnormality-detection'],kernel_sources=[],model_sources=[],
                  docker_image=parent['docker_image'])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Built CPU-only full public selected-series diagnostic',folder)
