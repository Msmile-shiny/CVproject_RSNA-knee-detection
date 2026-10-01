"""Submit an audited code version once; never submit the three visible rows as CSV."""
import argparse
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('candidate', choices=['fourway', 'cnx10'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--version', type=int, default=1)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    record = ROOT/f'{args.candidate}_submission_v{args.version}.json'
    if record.exists():
        print(record.read_text(encoding='utf-8'))
        return
    if args.candidate == 'fourway':
        r = json.loads((args.output/'sprint_fourway_receipt.json').read_text(encoding='utf-8'))
        assert r['ready_for_scoring'], r
        kernel = 'easoncyy/rsna-sprint-public-fourway'
        message = 'Public four-reader upgrade: exact upstream inference, 4/4 CoAt, no degraded visible branches; upstream claim 0.943 not 0.946'
    else:
        r = json.loads((args.output/'sprint_cnx10_receipt.json').read_text(encoding='utf-8'))
        assert r['status'] == 'COMPLETE' and r['parent']['ready_for_scoring'], r
        assert r['member_weight'] == .10
        kernel = 'easoncyy/rsna-sprint-cnx10'
        message = 'Four-reader public base + fixed 10% public ConvNeXt DINOv3 M448 fold0; no per-class tuning'
    api = KaggleApi(); api.authenticate()
    status = api.kernels_status(kernel)
    assert str(status.status).upper().endswith('COMPLETE'), status
    if not args.submit:
        print('Audit passed. Use --submit for hidden rerun:', kernel, args.version)
        return
    # Write an attempt marker BEFORE submitting, so an interrupted response cannot
    # silently cause a duplicate submission on a later invocation.
    attempt = ROOT/f'{args.candidate}_submit_attempt_v{args.version}.json'
    if attempt.exists():
        raise RuntimeError('Previous attempt needs reconciliation against submissions before retrying')
    attempt.write_text(json.dumps({'kernel':kernel,'version':args.version,'message':message},indent=2),encoding='utf-8')
    response = api.competition_submit_code(file_name='submission.csv', message=message,
        competition='rsna-knee-abnormality-detection', kernel=kernel, kernel_version=args.version)
    payload = response.to_dict() if hasattr(response,'to_dict') else str(response)
    record.write_text(json.dumps({'kernel':kernel,'version':args.version,'response':payload,'visible_receipt':r},indent=2,default=str),encoding='utf-8')
    print(record.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
