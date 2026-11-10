"""Fresh J64 preparation uses validated active queries, never historical sample counts."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import bridge_sampling_replay as bridge
import smol_sampling_replay as smol
import ral_confirm_analysis as confirm


class PrepareBoundariesTests(unittest.TestCase):
    def sources(self, root):
        code = root / 'checkout'; (code / 'code').mkdir(parents=True)
        (code / 'code/source.py').write_text('# immutable fixture source\n')
        return code, ('code/source.py',)

    def bridge_input(self, root):
        raw = root / 'raw'; rows = []; samples = []; ids = []
        for index in range(2):
            identity = f'data/simpler/confirm/shard00/t{index}_e0_r0_occluder'
            directory = raw / identity; directory.mkdir(parents=True)
            (directory.parent / 'server.json').write_text(json.dumps(dict(checkpoint_id='ckpt')))
            actions = np.random.default_rng(index).normal(size=(8, 4, 7)).astype(np.float32)
            np.savez(directory / 'q0005.npz', action_samples=actions, handoff=np.ones(2048, dtype=np.float16),
                     agent_image=np.zeros((256, 256, 3), dtype=np.uint8), state=np.zeros(8, dtype=np.float32))
            samples.append(actions); ids.append(str(directory))
            rows.append(dict(id=identity, dir=str(directory), artifact='q0005.npz', step=20,
                             task=index, failure=index, seed=3, device='GPU'))
        data = dict(ids=ids, eligible=True, query_step=np.array([20, 20]), task=np.arange(2), y=np.arange(2),
                    scene=np.zeros(2,dtype=int), replica=np.zeros(2,dtype=int),
                    Spread=np.asarray([bridge.spread_features(value)[0] for value in samples]))
        return raw, rows, data

    def test_bridge_two_query_prepare_separate_code_root_and_hash_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); code, sources = self.sources(root); raw, rows, data = self.bridge_input(root)
            out = root / 'prepared'
            with patch.object(bridge, 'ROOT', raw), patch.object(bridge, 'CODE_ROOT', code), \
                    patch.object(bridge, 'SOURCES', sources), \
                    patch.object(bridge, 'load', return_value=({(bridge.CELL, bridge.STEP):data}, {})), \
                    patch.object(bridge.replay, 'cohort', return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                bridge.prepare(out)
                receipt, _ = bridge.receipt(out)
                self.assertEqual(receipt['queries'], 2)
                self.assertEqual(len(receipt['rows']), 2)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    bridge.prepare(out)
                artifact = Path(rows[0]['dir']) / rows[0]['artifact']
                artifact.write_bytes(artifact.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    bridge.read_input(receipt['rows'][0])
                (out / 'dataset.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Prepared dataset changed'):
                    bridge.receipt(out)

    def test_bridge_duplicate_and_drifted_identities_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw, rows, data = self.bridge_input(root)
            for bad in ([rows[0], rows[0]], [dict(rows[0], id='data/different'), rows[1]]):
                with patch.object(bridge, 'ROOT', raw), \
                        patch.object(bridge, 'load', return_value=({(bridge.CELL, bridge.STEP):data}, {})), \
                        patch.object(bridge.replay, 'cohort', return_value=bad):
                    with self.assertRaises(ValueError):
                        bridge.prepare(root / 'must-not-exist')
                self.assertFalse((root / 'must-not-exist').exists())

    def smol_input(self, root):
        raw = root / 'raw'; datasets = {}; rows = {}
        metadata = {field:'same' for field in smol.SERVER_FIELDS}
        metadata.update(checkpoint_id='ckpt', device='GPU')
        for index, cell in enumerate(smol.CELLS):
            identity = f'data/{cell}/shard00/t0_e15_r0_occluder'; directory = raw / identity
            directory.mkdir(parents=True)
            server = directory.parent / 'server.json'; server.write_text(json.dumps(metadata))
            image = np.zeros((256, 256, 3), dtype=np.uint8); state = np.zeros(8,dtype=np.float32)
            actions = np.random.default_rng(index).normal(size=(8,10,7)).astype(np.float32)
            handoff = np.ones(10240,dtype=np.float16); path = directory / 'q0008.npz'
            np.savez_compressed(path, agent_image=image, wrist_image=image, state=state,
                                action_samples=actions, handoff=handoff, executed_actions=actions[0])
            import hashlib
            image_hash = hashlib.sha256(str((image.shape,str(image.dtype))).encode()+image.tobytes()).hexdigest()
            rows[cell] = [dict(id=identity, cell=cell, artifact=path.name, artifact_sha256=smol.digest(path),
                              step=80, task=0, scene=15, replica=0, failure=index, seed=3,
                              independent_seed=smol.independent_seed(identity), checkpoint='ckpt', device='GPU',
                              server=server.relative_to(raw).as_posix(), state=state.tolist(),
                              image_sha256=[image_hash,image_hash])]
            datasets[cell] = dict(ids=[str(directory)],eligible=True,task=np.array([0]),scene=np.array([15]),
                                  replica=np.array([0]),y=np.array([index]),query_step=np.array([80]),
                                  Spread=np.array([smol.spread_features(actions)[0]]),H=handoff[None].astype(np.float64))
        return raw, rows, datasets

    def test_smol_one_query_per_suite_prepares_and_artifact_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code,sources=self.sources(root);raw,rows,datasets=self.smol_input(root)
            def load(_,cell,reader):return {(cell,smol.STEP):datasets[cell]},{}
            with patch.object(smol,'ROOT',raw),patch.object(smol,'CODE_ROOT',code),patch.object(smol,'SOURCES',sources), \
                    patch.object(confirm,'load',side_effect=load),patch.object(smol,'cohort',side_effect=lambda cell:rows[cell]), \
                    contextlib.redirect_stdout(io.StringIO()):
                smol.prepare(root/'prepared')
                receipt,_=smol.receipt(root/'prepared')
                self.assertEqual(receipt['queries'],2)
                row=rows[smol.CELLS[0]][0]; artifact=raw/row['id']/row['artifact']
                artifact.write_bytes(artifact.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'Input digest mismatch'):
                    smol.read_input(row)

    def test_smol_duplicate_and_drifted_identities_are_boundary_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,rows,datasets=self.smol_input(root)
            def load(_,cell,reader):return {(cell,smol.STEP):datasets[cell]},{}
            first=rows[smol.CELLS[0]][0]
            for bad in ([first,first], [dict(first,id='data/different')]):
                with patch.object(smol,'ROOT',raw),patch.object(confirm,'load',side_effect=load), \
                        patch.object(smol,'cohort',return_value=bad):
                    with self.assertRaises(ValueError):smol.prepare(root/'must-not-exist')
                self.assertFalse((root/'must-not-exist').exists())
