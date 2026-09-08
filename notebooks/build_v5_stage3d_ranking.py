"""Generate matched 130-mm old-v5 controls and auxiliary ranking experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABEL_SHA = 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Source contract drift: ' + old[:90])
    return source.replace(old, new, 1)


def build(rank_lambda):
    trust_dir = ROOT.parent / 'data/processed/stage3d_trust'
    trust_manifest = json.loads((trust_dir / 'manifest.json').read_text())
    asset_sha = hashlib.sha256((trust_dir / 'stage3d_trust.npz').read_bytes()).hexdigest()
    assert asset_sha == trust_manifest['asset_sha256']
    assert hashlib.sha256((ROOT.parent / 'data/processed/v5_labels.csv').read_bytes()).hexdigest() == LABEL_SHA
    tag = 'stage3d_rank' if rank_lambda else 'stage3d_control'
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        f'# {tag}\n130mm / historical v5 labels / seed42. Auxiliary ranking lambda={rank_lambda}.\n',
        'Mount competition, rsna-dinov2-weights, rsna-knee-v5-labels and rsna-knee-stage3d-trust.\n',
        'Rule-selected candidates are not clinical ground truth. Gold is a reused development set.\n']}]
    for path in sorted((ROOT / 'cells_v5').iterdir()):
        if path.suffix not in ('.py', '.md'):
            continue
        s = path.read_text(encoding='utf-8')
        if path.name == '03_config.py':
            s += f"\nCFG.update(experiment_name={tag!r}, rank_lambda={rank_lambda}, rank_margin=0.1, rank_min_cases=20, trust_input='/kaggle/input/datasets/easoncyy/rsna-knee-stage3d-trust')\n"
            s += """
# Resolve documented old/new Kaggle mount layouts; missing weights must stop.
def resolve_asset(config_key, filename, dataset_slug):
    configured = Path(CFG[config_key])
    candidates = [configured / filename if config_key != 'dinov2_weights' else configured,
                  Path('/kaggle/input') / dataset_slug / filename,
                  Path('/kaggle/input/datasets/easoncyy') / dataset_slug / filename]
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(f'Mount {dataset_slug}/{filename}; checked {candidates}')
    CFG[config_key] = str(found if config_key == 'dinov2_weights' else found.parent)
resolve_asset('label_input', 'v5_labels.csv', 'rsna-knee-v5-labels')
resolve_asset('trust_input', 'stage3d_trust.npz', 'rsna-knee-stage3d-trust')
resolve_asset('dinov2_weights', 'dinov2_vits14.pth', 'rsna-dinov2-weights')
print('Stage 3D configuration:', CFG)
"""
        elif path.name == '07_loss.py':
            s += '\n' + (ROOT / 'stage3d_rank_loss.py').read_text()
        elif path.name == '08_dataset.py':
            s = replace_once(s, "'prob_targets': probs, 'weights': weights, 'soft_masks': soft_masks,", "'prob_targets': probs, 'weights': weights, 'soft_masks': soft_masks,\n                'rank_states': torch.tensor([label_row['rank_' + c] for c in TARGET_COLUMNS], dtype=torch.int8),")
        elif path.name == '09_load_data.py':
            s = replace_once(s, 'fused_df = pd.read_csv(v5_label_file)', f"import hashlib, json\nassert hashlib.sha256(v5_label_file.read_bytes()).hexdigest() == {LABEL_SHA!r}, 'Wrong historical v5 labels'\nfused_df = pd.read_csv(v5_label_file)")
            s += f'''
_trust_path = Path(CFG['trust_input']) / 'stage3d_trust.npz'
assert hashlib.sha256(_trust_path.read_bytes()).hexdigest() == {asset_sha!r}, 'Wrong ranking asset'
_trust = np.load(_trust_path, allow_pickle=False)
assert list(_trust['targets']) == TARGET_COLUMNS
assert _trust['state'].shape == (len(_trust['ids']), len(TARGET_COLUMNS))
assert np.isin(_trust['state'], [-1, 0, 1]).all()
_rank_df = pd.DataFrame(_trust['state'], index=_trust['ids'].astype(str), columns=TARGET_COLUMNS)
assert _rank_df.index.is_unique
assert set(_rank_df.index) == set(unlabeled_studies) == set(train_labels.index)
assert not set(_rank_df.index) & set(val_labels.index)
_active_classes = []
for _c in TARGET_COLUMNS:
    _counts = _rank_df[_c].value_counts()
    if min(_counts.get(1, 0), _counts.get(-1, 0)) < CFG['rank_min_cases']:
        _rank_df[_c] = 0
    else:
        _active_classes.append(_c)
    train_labels['rank_' + _c] = _rank_df[_c].reindex(train_labels.index)
assert np.isfinite(train_labels[PROB_COLS + WEIGHT_COLS + MASK_COLS].values).all()
_preflight = {{'config': CFG, 'label_sha256': {LABEL_SHA!r}, 'trust_sha256': {asset_sha!r},
              'n_train': len(train_labels), 'n_gold': len(val_labels), 'rank_classes': _active_classes,
              'teacher_provenance': 'Crossfit training manifest unavailable; not independently verified'}}
Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)
(Path(CFG['output_dir']) / 'phase3d_preflight.json').write_text(json.dumps(_preflight, indent=2))
print('Ranking classes:', _active_classes)
'''
        elif path.name == '12_train_val.py':
            s = replace_once(s, "    grad_accum = CFG.get('grad_accum_steps', 1)", "    grad_accum = CFG.get('grad_accum_steps', 1)\n    rank_total = bce_total = active_batches = 0\n    pair_counts = np.zeros(len(TARGET_COLUMNS), dtype=np.int64)")
            s = replace_once(s, '            loss = criterion(logits, prob_targets, weights, soft_masks)', '''            bce_loss = criterion(logits, prob_targets, weights, soft_masks)
            rank_loss, counts = trusted_pair_rank_loss(
                logits, batch['rank_states'].to(DEVICE), soft_masks, CFG['rank_margin'])
            loss = bce_loss + CFG['rank_lambda'] * rank_loss
            bce_total += bce_loss.detach().item()
            rank_total += rank_loss.detach().item()
            pair_counts += np.asarray(counts)
            active_batches += int(sum(counts) > 0)''')
            s = replace_once(s, '    return total_loss / max(n_batches, 1)', '''    record = {'epoch': epoch, 'bce': bce_total / max(n_batches, 1),
              'rank_loss': rank_total / max(n_batches, 1),
              'weighted_rank_loss': CFG['rank_lambda'] * rank_total / max(n_batches, 1),
              'active_batches': active_batches, 'batches': n_batches,
              'pairs_by_class': dict(zip(TARGET_COLUMNS, pair_counts.tolist()))}
    with (Path(CFG['output_dir']) / 'ranking_history.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\\n')
    print('Ranking diagnostics:', record)
    return total_loss / max(n_batches, 1)''')
        elif path.name == '15_training_loop.py':
            s = "(Path(CFG['output_dir']) / 'ranking_history.jsonl').write_text('')\n" + s
        elif path.name == '17_submission.py':
            s += "\n_phase3d_final = dict(_preflight, gold_macro_auc=gold_macro, best_epoch=best_epoch)\n(output_dir / 'phase3d_manifest.json').write_text(json.dumps(_phase3d_final, indent=2))\n"
        if path.suffix == '.py':
            compile(s, path.name, 'exec')
            cells.append({'cell_type': 'code', 'metadata': {}, 'source': s.splitlines(True), 'outputs': [], 'execution_count': None})
        else:
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)})
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': cells}
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:02d}'
    output = ROOT / f'kaggle_train_v5_{tag}.ipynb'
    output.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    build(0.0)
    build(0.05)
