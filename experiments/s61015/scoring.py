"""Read-only scoring reconciliation; no retry, launch or final selection."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
sys.path.insert(0,str(HERE.parent/'r71014'))
from run_scale_probe import prepare_dns
from refresh_finalists import classify
from kaggle.api.kaggle_api_extended import KaggleApi
receipt=json.loads((HERE/'submission.json').read_text())
prepare_dns();api=KaggleApi();api.authenticate()
rows=[s.to_dict() for s in api.competition_submissions('rsna-knee-abnormality-detection')]
row=next(s for s in rows if int(s['ref'])==receipt['response']['ref'])
assert row['url'].startswith('/code/'+receipt['kernel']+'?scriptVersionId=')
state,score=classify(row)
record=dict(checked_at=datetime.now(timezone.utc).isoformat(),state=state,public_score=score,
            protected_submission=57144024,protected_public=.956,raw=row,final_selection_changed=False)
(HERE/'scoring.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
