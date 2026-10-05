# Appended to author's definitions in an isolated process. Never import this file.
import importlib.util
import sys

cfg = __PIXEL_CONFIG__
adopt_config_globals(cfg)
test = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
series = pd.read_csv(ROOT / 'test_series.csv', dtype=str)
plane_map = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
headers = annotate(walk('test_series'))
slot_map = pick_slots(headers, plane_map)
studies, cache, mask = build_cache(slot_map, plane_map, lat_of(headers), 'meniscus10')
assert set(studies) == set(test.StudyInstanceUID), 'Incomplete study coverage'
assert not DECODE_FAILED, 'Failed decodes: no degraded submission allowed'
assert all('ordered' in rec for slots in slot_map.values() for rec in slots.values()), 'Ordering interrupted'
assert int(mask.sum()) == sum(len(slots) for slots in slot_map.values()), 'Incomplete slot decode'
assert np.all(mask.sum(axis=1) > 0), 'Empty study'
roots = [p.parent for p in Path('/kaggle/input').rglob('bundle_manifest.json')
         if json.loads(p.read_text()).get('schema_version') == 'public0033_meniscus10_bundle_v1']
assert len(roots) == 1, 'Bundle must resolve uniquely'
runtime_path = roots[0] / 'public0033_runtime.py'
assert hashlib.sha256(runtime_path.read_bytes()).hexdigest() == '9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683'
spec = importlib.util.spec_from_file_location('public0033_runtime', runtime_path)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)
runtime.verify_bundle(roots[0])
work = Path('/kaggle/working/meniscus10')
work.mkdir(exist_ok=True)
preprocess_seconds = time.time() - T0
for device in range(torch.cuda.device_count()):
    # The pinned PyTorch runtime requires an allocator on this device before
    # reset_peak_memory_stats; device_count alone does not initialize it.
    with torch.cuda.device(device):
        allocator_warmup = torch.empty(1, device=f'cuda:{device}')
        del allocator_warmup
    torch.cuda.reset_peak_memory_stats(device)
# Exercise the full hidden-run batch before any parent inference. Repeated visible
# inputs are an engineering measurement only and never enter final predictions.
assert torch.cuda.device_count() == 2
stress_model = runtime._load_fullfit_model(roots[0], torch.device('cuda:0'))
stress_indices = np.arange(runtime.BATCH_STUDIES) % len(studies)
stress_started = time.monotonic()
with torch.inference_mode():
    stress_images = torch.from_numpy(np.ascontiguousarray(cache[stress_indices, :, :3])).to('cuda:0')
    stress_mask = torch.from_numpy(np.ascontiguousarray(mask[stress_indices])).to('cuda:0')
    stress_values = stress_model(stress_images, stress_mask)
    assert stress_values.shape == (runtime.BATCH_STUDIES, 12) and torch.isfinite(stress_values).all()
    torch.cuda.synchronize(0)
stress_seconds = time.monotonic() - stress_started
stress_peak = torch.cuda.max_memory_allocated(0)
del stress_model, stress_images, stress_mask, stress_values
gc.collect()
torch.cuda.empty_cache()
result = runtime.run_cached_inference(cache, mask, studies, {
    'bag_raw_csv': str(work / 'public0033_bag_raw.csv'),
    'receipt_json': str(work / 'public0033_cached_inference_receipt.json'),
    'work_dir': str(work),
})
assert result['status'] == 'passed' and result['fallback'] == 0
receipt = dict(status='COMPLETE', study_count=len(studies), preprocess_seconds=preprocess_seconds,
               elapsed_seconds=time.time()-T0, source_sha256='__SOURCE_SHA__',
               pixel_config=cfg, slots_present=mask.sum(1).tolist(),
               peak_gpu_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
               full_batch_stress_pass=True, stress_batch_studies=runtime.BATCH_STUDIES,
               stress_seconds=stress_seconds, stress_peak_gpu_bytes=stress_peak,
               decode_failures=0, partial_predictions=0)
(work / 'specialist_receipt.json').write_text(json.dumps(receipt, indent=2))
print('MENISCUS SPECIALIST COMPLETE', receipt, flush=True)
