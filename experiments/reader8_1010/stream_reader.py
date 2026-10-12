"""Prepare lower-live-input-memory reader; no GPU performance claim."""
import ast,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def optimize(src):
    old='        x1, mask1, pos1 = x1.cuda(non_blocking=True), mask1.cuda(), pos1.cuda()\n'
    assert src.count(old)==1;src=src.replace(old,'')
    old='        x1_flip = torch.flip(x1, dims=[-1])\n'
    assert src.count(old)==1;src=src.replace(old,'')
    marker='            # Stride 1 Multi-Scale (256px + 320px) with Bilateral TTA'
    assert src.count(marker)==1
    src=src.replace(marker,'''            del x0, x0_flip, mask0, pos0
            x1, mask1, pos1 = x1.cuda(non_blocking=True), mask1.cuda(), pos1.cuda()
            x1_flip = torch.flip(x1, dims=[-1])
'''+marker)
    marker='        out.append(p.cpu())'
    assert src.count(marker)==1
    src=src.replace(marker,'        del x1, x1_flip, mask1, pos1\n'+marker)
    ast.parse(src)
    return src

if __name__=='__main__':
    nb=json.loads((HERE/'candidate/reader8.ipynb').read_bytes())
    src=''.join(nb['cells'][9]['source']).split('\n',1)[1]
    improved=optimize(src)
    # Verify all model calls and numerical expressions are unchanged.
    def expressions(s):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.BinOp,ast.Call)) and not (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cuda')]
    assert sorted(expressions(src))==sorted(expressions(improved))
    (HERE/'stream_infer.py').write_text(improved,encoding='utf8')
    result=dict(status='PREPARED_NOT_GPU_TESTED',original_sha256=hashlib.sha256(src.encode()).hexdigest(),optimized_sha256=hashlib.sha256(improved.encode()).hexdigest(),numeric_expressions_unchanged=True,live_full_image_tensors_before=4,live_full_image_tensors_after=2)
    (HERE/'stream_build.json').write_text(json.dumps(result,indent=2));print(result)
