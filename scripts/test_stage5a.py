"""Behavioral checks for masking, permutation invariance and trainability."""
import sys
import ast
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from stage5a_core import DenseMIL, window_centers, selected_tokens, weighted_bce

class TestStage5A(unittest.TestCase):
    def runtime_functions(self, names, env):
        path=Path(__file__).resolve().parents[1]/'notebooks/stage5a_runtime.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        exec(compile(functions,'runtime-functions','exec'),env)

    def test_resume_provenance(self):
        import copy
        env={}; self.runtime_functions({'compatible_run'},env)
        current=dict(experiment='stage5a',checkpoint_sha256='weights',labels_sha256='labels',
                     config=dict(coarse_px=168,fine_px=280,slices=32,top_per_class=2,seed=42,epochs=20,head_lr=.0003,batch_size=32))
        old=copy.deepcopy(current)
        old['config'].update(feature_minutes=300,resume_input='old-path')
        env['compatible_run'](old,current)
        for key in ('checkpoint_sha256','labels_sha256'):
            bad=copy.deepcopy(old); bad[key]='different'
            with self.assertRaises(ValueError): env['compatible_run'](bad,current)
        bad=copy.deepcopy(old); bad['config']['fine_px']=336
        with self.assertRaises(ValueError): env['compatible_run'](bad,current)

    def test_cache_recovery_and_corruption(self):
        import hashlib,shutil,zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); previous=root/'previous'; current=root/'current'
            env=dict(np=np,Path=Path,hashlib=hashlib,shutil=shutil,zipfile=zipfile,
                     output_dir=current,resume_roots=[previous],N_SLOT=6,S5=dict(slices=32))
            self.runtime_functions({'cache_path','validated_cache','restore_cache'},env)
            dest=env['cache_path']('case-1','fine'); source=previous/'stage5a_features/fine'/dest.name
            source.parent.mkdir(parents=True)
            bank=dict(x=np.ones((2,1152),np.float16),mask=np.ones(2,bool),slot=np.array([0,2],np.int64),
                      pos=np.array([.2,.7],np.float32),center=np.array([3,8],np.int64),scale=np.ones(2,np.int64))
            np.savez_compressed(source,**bank)
            self.assertTrue(env['restore_cache']('case-1','fine',[(0,3),(2,8)]))
            self.assertEqual(source.read_bytes(),dest.read_bytes())
            self.assertFalse(env['validated_cache'](source,'fine',[(0,3),(2,9)]))
            source.write_bytes(b'truncated ZIP')
            self.assertFalse(env['restore_cache']('case-1','fine',[(0,3),(2,8)]))

    def test_pause_preserves_coarse_exports_and_resumes(self):
        import json,time
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); calls=[]
            def coarse(*args,**kwargs): return torch.zeros(1,12),torch.ones(1,12,2)
            env=dict(torch=torch,np=np,json=json,time=time,Path=Path,output_dir=root,
                     session_started=time.monotonic(),audit={},resume_roots=[],
                     all_uids=['a','b'],train_uids=['a'],gold_uids=['b'],test_uids=[],
                     maps={},labels=None,S5=dict(feature_minutes=300,top_per_class=2,smoke_studies=0),
                     session_budget_reached=Mock(side_effect=[False]*6+[True]),
                     restore_cache=lambda *args:True,
                     extract_study=Mock(side_effect=AssertionError('Cached cases must not decode')),
                     get_head=lambda *args,**kwargs:coarse,
                     export_head=lambda name,*args:calls.append(name),
                     load_bank=lambda uid:dict(mask=np.ones(2,bool),slot=np.array([0,2]),center=np.array([3,8])),
                     tensor_batch=lambda ids:[],selected_tokens=selected_tokens)
            self.runtime_functions({'atomic_json','run_stage5a'},env)
            env['run_stage5a']()
            manifest=json.loads((root/'stage5a_manifest.json').read_text())
            self.assertEqual(manifest['status'],'paused'); self.assertFalse(manifest['completed'])
            self.assertEqual(manifest['stage'],'fine'); self.assertEqual(manifest['processed'],1)
            self.assertEqual(calls,['mean','coarse','coarse_continue'])
            self.assertEqual(len(json.loads((root/'stage5a_selected_slices.json').read_text())),1)
            # Resume to completion with no cache recomputation or lost controls.
            env['session_budget_reached']=lambda:False
            env['run_stage5a']()
            manifest=json.loads((root/'stage5a_manifest.json').read_text())
            self.assertTrue(manifest['completed']); self.assertEqual(manifest['reused'],dict(coarse=2,fine=2))
            self.assertEqual(calls[-1],'fine')

    def test_real_backbone_resize_and_forward(self):
        import timm, math
        torch.set_num_threads(2)
        source=(Path(__file__).resolve().parents[1]/'notebooks/stage5a_runtime.py').read_text(encoding='utf-8')
        original=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=0,img_size=288)
        env=dict(timm=timm,torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,
                 backbone_state=original.state_dict(),S5=dict(coarse_px=168,fine_px=280),
                 CFG=dict(dinov2_variant='vit_small_patch14_dinov2.lvd142m'),DEVICE=torch.device('cpu'),N_GPUS=0)
        exec(source[source.index('encoders = {}'):source.index('audit = dict(')],env)
        for size,model in env['encoders'].items():
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            with torch.no_grad(): out=model.forward_features(torch.zeros(1,3,size,size))
            self.assertEqual(tuple(out.shape),(1,(size//14)**2+1,384))

    def test_centers_are_unique_real_adjacent_windows(self):
        for n in (0,1,2,3,9,32,100):
            c=window_centers(n)
            self.assertEqual(len(c),len(set(c)))
            self.assertLessEqual(len(c),32)
            if len(c): self.assertTrue(((c>=1)&(c<n-1)).all())

    def test_mask_and_permutation(self):
        torch.manual_seed(7)
        model=DenseMIL(dim=8,hidden=16).eval()
        x=torch.randn(2,9,8); mask=torch.ones(2,9,dtype=torch.bool); mask[:,-2:]=False
        slots=torch.zeros(2,9,dtype=torch.long); pos=torch.rand(2,9); scale=slots.clone()
        y,a=model(x,mask,slots,pos,scale,True)
        x[:,-2:]=10000
        self.assertTrue(torch.allclose(y,model(x,mask,slots,pos,scale)))
        p=torch.randperm(9)
        self.assertTrue(torch.allclose(y,model(x[:,p],mask[:,p],slots[:,p],pos[:,p],scale[:,p]),atol=1e-6))
        self.assertEqual(float(a[:,:,-2:].abs().sum().detach()),0.)
        self.assertTrue(torch.isfinite(model(x,mask*False,slots,pos,scale)).all())

    def test_selection_and_loss(self):
        idx=selected_tokens(np.ones((12,32)),np.arange(32)<7,2)
        self.assertTrue((idx<7).all()); self.assertLessEqual(len(idx),24)
        logits=torch.randn(4,12,requires_grad=True)
        loss=weighted_bce(logits,torch.rand(4,12),torch.ones(4,12),torch.ones(4,12))
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(weighted_bce(logits,torch.rand(4,12),torch.ones(4,12),torch.zeros(4,12)).detach()),0.)

    def test_feature_training_pipeline(self):
        import pandas as pd
        import hashlib
        torch.set_num_threads(2)
        source=(Path(__file__).resolve().parents[1]/'notebooks/stage5a_runtime.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        names={'cache_path','load_bank','tensor_batch','fit_head','predict_head'}
        functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
        with tempfile.TemporaryDirectory() as tmp:
            columns=[[f'{prefix}{j}' for j in range(12)] for prefix in ('p','w','m')]
            env=dict(torch=torch,nn=torch.nn,np=np,pd=pd,Path=Path,hashlib=hashlib,
                     output_dir=Path(tmp),BANKS={},DEVICE=torch.device('cpu'),DenseMIL=DenseMIL,weighted_bce=weighted_bce,
                     S5=dict(seed=42,epochs=1,head_lr=.001,batch_size=2),audit={},
                     PROB_COLS=columns[0],WEIGHT_COLS=columns[1],MASK_COLS=columns[2])
            exec(compile(functions,'runtime-functions','exec'),env)
            rng=np.random.default_rng(2)
            for uid,n in [('a',4),('b',7)]:
                p=env['cache_path'](uid,'coarse'); p.parent.mkdir(exist_ok=True,parents=True)
                np.savez(p,x=rng.normal(size=(n,1152)).astype(np.float16),mask=np.ones(n,bool),
                         slot=np.zeros(n,np.int64),scale=np.zeros(n,np.int64),pos=np.linspace(0,1,n,dtype=np.float32),center=np.arange(n))
            labels=pd.DataFrame(np.ones((2,36),np.float32),index=['a','b'],columns=sum(columns,[]))
            head=env['fit_head'](['a','b'],labels,'test')
            pred=env['predict_head'](head,['b','a'])
            self.assertEqual(pred.shape,(2,12)); self.assertTrue(np.isfinite(pred).all())
            single=env['predict_head'](head,['a'])
            self.assertTrue(np.allclose(single[0],pred[1],atol=1e-6))

if __name__=='__main__': unittest.main()
