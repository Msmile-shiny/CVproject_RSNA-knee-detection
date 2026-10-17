"""Create a local editorial draft without changing scored executable cells."""
import hashlib,json
from pathlib import Path
import nbformat
HERE=Path(__file__).resolve().parent
raw=(HERE/'candidate/s6.ipynb').read_bytes()
build=json.loads((HERE/'build.json').read_text())
assert hashlib.sha256(raw).hexdigest()==build['source_sha256']
nb=nbformat.reads(raw.decode(),as_version=4)
before=[c.source for c in nb.cells if c.cell_type=='code']
nb.cells.insert(0,nbformat.v4.new_markdown_cell((HERE/'publication_note.md').read_text(encoding='utf8')))
assert before==[c.source for c in nb.cells if c.cell_type=='code']
nbformat.validate(nb)
out=HERE/'publication';out.mkdir(exist_ok=True)
nbformat.write(nb,out/'r7_s6_technical_report.ipynb')
(out/'verification.json').write_text(json.dumps(dict(status='LOCAL_DRAFT',code_cells_unchanged=True,scored_source_sha256=build['source_sha256'],publication_sha256=hashlib.sha256((out/'r7_s6_technical_report.ipynb').read_bytes()).hexdigest(),published=False,license_review_complete=False),indent=2))
print('Local publication draft validated; executable cells unchanged; not uploaded.')
