"""Single-factor removal of the hand-specified cross-finding rank adjustment."""
import argparse,ast,copy,hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    raw=(ROOT/'experiments/swa2241010/upstream/parent950.ipynb').read_bytes()
    assert sha(raw)=='16ce7c78f860b6d07331c8d39c6a65414cde200462a64c984b04bd8efc167289'
    parent=json.loads(raw);nb=copy.deepcopy(parent)
    source=''.join(nb['cells'][9]['source'])
    start=source.index('        for name, peers in PEERS.items():')
    end=source.index('        # Final percentile rank normalization',start)
    removed=source[start:end]
    assert '0.98 * tier3_df[name] + 0.02 * peer_rank' in removed
    source=source[:start]+'        # Controlled ablation: no cross-finding adjustment.\n\n'+source[end:]
    # Force a successful full parent recipe, refusing any earlier tier fallback.
    source='_nopeer_success=False\n'+source
    marker='        tier3_df.to_csv(_o_pub, index=False)'
    assert source.count(marker)==1;source=source.replace(marker,marker+'\n        _nopeer_success=True')
    nb['cells'][9]['source']=source
    assert all(nb['cells'][i]==parent['cells'][i] for i in range(len(parent['cells'])) if i!=9)
    reader=''.join(parent['cells'][8]['source']).split('\n',1)[1]
    audited=json.loads((ROOT/'experiments/reader8_1010/candidate/reader8.ipynb').read_bytes())
    guard=copy.deepcopy(audited['cells'][0])
    audit="assert _nopeer_success, 'No-peer full fusion failed; refusing fallback'\n"+''.join(audited['cells'][-1]['source'])
    nb['cells']=[guard]+nb['cells']+[nbformat.v4.new_code_cell(audit)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'nopeer-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    folder=HERE/'candidate';folder.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode();(folder/'reader8.ipynb').write_bytes(payload)
    meta=json.loads((ROOT/'experiments/reader8_1010/candidate/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-parent950-nopeer',title='RSNA Parent950 NoPeer')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    built=dict(kernel=meta['id'],output_prefix='nopeer',source_sha256=sha(payload),reader_sha256=sha((reader.rstrip('\n')+'\n').encode()),parent_sha256=sha(raw),changed_parent_cells=[9],image_pipeline_unchanged=True,use_peers=False,submission_message='0.950 parent single-factor ablation: disable hand-specified disease peer graph; all image predictions and class fusion weights unchanged')
    (HERE/'build.json').write_text(json.dumps(built,indent=2));print(built)

def shared():
    spec=importlib.util.spec_from_file_location('reader_shared',ROOT/'experiments/reader8_1010/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE
    return module

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    if args.action=='build':build()
    else:shared().online(args.action,args.submit)
