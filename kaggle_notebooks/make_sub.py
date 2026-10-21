"""Generate a submission kernel dir from sub_template.py.

Blend weights = L-BFGS NLL fit on the full 5-fold OOF of each component (results/oof_<oof>.npy).
usage: python kaggle_notebooks/make_sub.py <version-name> "<title>"     (component set edited below)
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from common import RESULTS, load  # noqa: E402
from e05_blend import fit_weights_nll  # noqa: E402

FEATS = {  # research kernels providing train embeddings
    'q8b': dict(kernel='ntl-feat-qwen3-8b', hidden=4096, nlayers=36, C=0.0025, drop_bos=False,
                model='qwen-lm/qwen-3/transformers/8b-base/1', dataset=None),
    'q35': dict(kernel='ntl-feat-qwen35-9b', hidden=4096, nlayers=32, C=0.0025, drop_bos=True,
                model=None, dataset='bbucxi/qwen35-9b-safetensors'),
}
LORA_KERNELS = {  # name -> (members [(kernel, adapter subdir)], base model)
    'lora_q8b': ([('ntl-lora-q8b', 'adapter')], 'qwen-lm/qwen-3/transformers/8b-base/1'),
    'lora_q8b_x': ([('ntl-lora-q8b-extra2', 'adapter'), ('ntl-lora-q8b-extra2', 'fold2'),
                    ('ntl-lora-q8b-extra', 'fold0'), ('ntl-lora-q8b-extra', 'fold1')], 'qwen-lm/qwen-3/transformers/8b-base/1'),
    'lora_q8b_pool': ([('ntl-lora-q8b-pool', 'adapter'), ('ntl-lora-q8b-pool', 'fold0')], 'qwen-lm/qwen-3/transformers/8b-base/1'),
    'lora_q8b_twin': ([('ntl-lora-q8b-twin', 'adapter'), ('ntl-lora-q8b-twin', 'fold0')], 'qwen-lm/qwen-3/transformers/8b-base/1'),
    'lora_q8b_x6': ([('ntl-lora-q8b-extra2', 'adapter'), ('ntl-lora-q8b-extra2', 'fold2'), ('ntl-lora-q8b-extra', 'fold0'),
                     ('ntl-lora-q8b-extra', 'fold1'), ('ntl-lora-q8b-pool', 'adapter'), ('ntl-lora-q8b-pool', 'fold0')],
                    'qwen-lm/qwen-3/transformers/8b-base/1'),
}


def build(name, title, components, extra_w=0.0, claude_gate=0.0, sinkhorn=0.0, twin_w=0.0, head_extra_w=0.0, fit_idx=None, pool=False, heads_kernel='ntl-extra-heads', lora_twin_w=0.0):
    """components: list of (component_name, oof_name). component_name forms:
    'stylegram' | 'tfidf' | '<feat>_L<a+b>' | 'lora_*'"""
    tr = load()
    y = tr.y.values
    Ls = [np.log(np.clip(np.load(RESULTS / f'oof_{o}.npy'), 1e-9, 1)) for _, o in components]
    w = fit_weights_nll(Ls, y, np.arange(len(y)) if fit_idx is None else fit_idx)
    W = {c: round(float(x), 4) for (c, _), x in zip(components, w)}
    heads, models, datasets, kernels, loras = {}, set(), set(), set(), []
    for c, _ in components:
        if '_L' in c and not c.startswith('lora'):
            f, g = c.split('_L')
            spec = FEATS[f]
            h = heads.setdefault(f, dict(name=f, kernel=spec['kernel'], hidden=spec['hidden'], nlayers=spec['nlayers'],
                                         groups=[], C=spec['C'], drop_bos=spec['drop_bos']))
            if head_extra_w > 0:
                h.update(extra_w=head_extra_w, extra_kernel=heads_kernel, extra_layers={'q35': [17, 18], 'q8b': [2, 22]}[f])
                kernels.add('abhayganti09/' + heads_kernel)
            h['groups'].append([int(x) for x in g.split('+')])
            kernels.add('abhayganti09/' + spec['kernel'])
            if spec['model']:
                models.add(spec['model'])
            if spec['dataset']:
                datasets.add(spec['dataset'])
        elif c.startswith('lora'):
            mem, m = LORA_KERNELS[c]
            loras.append(dict(name=c, kernel=mem[0][0], members=mem, twin_w=lora_twin_w))
            kernels |= {'abhayganti09/' + k for k, _ in mem}
            models.add(m)
    s = (ROOT / 'kaggle_notebooks/sub_template.py').read_text()
    s = s.replace('__TITLE__', title).replace('__HEADS__', repr(list(heads.values()))) \
         .replace('__WEIGHTS__', repr(W)).replace('__LORAS__', repr(loras)) \
         .replace('__EXTRA_W__', repr(extra_w)).replace('__CLAUDE_GATE__', repr(claude_gate)).replace('__SINKHORN__', repr(sinkhorn)).replace('__TWIN_W__', repr(twin_w)).replace('__POOL__', repr(pool))
    if pool:
        exec(open(ROOT / 'kaggle_research/extra_pool_snippet.py').read(), globals())
        datasets |= set(EXTRA_DATASETS)
    if extra_w > 0 or claude_gate > 0:
        datasets |= {'seyitkaangunes/ntl-extra-api-passages', 'abdullah1afify/ntl-generations',
                     'keshavatearth/ntl-gpt-new-prompts-v1', 'prvsiyan/ntl-own-prompt-generations-v1'}
    d = ROOT / 'kaggle_notebooks' / name
    d.mkdir(exist_ok=True)
    (d / 'main.py').write_text(s)
    meta = {'id': f'abhayganti09/{name}', 'title': name, 'code_file': 'main.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': False,
            'dataset_sources': sorted(datasets), 'competition_sources': ['name-that-llm'],
            'kernel_sources': sorted(kernels), 'model_sources': sorted(models), 'machine_shape': 'NvidiaTeslaT4'}
    (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
    print('weights', W)
    print('wrote', d)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('title')
    ap.add_argument('--extra_w', type=float, default=0.0)
    ap.add_argument('--claude_gate', type=float, default=0.0)
    ap.add_argument('--sinkhorn', type=float, default=0.0)
    ap.add_argument('--twin_w', type=float, default=0.0)
    ap.add_argument('--head_extra_w', type=float, default=0.0)
    ap.add_argument('--heads_pool', action='store_true', help='frozen heads trained with the big pool (ntl-extra-heads-pool, w 0.5)')
    ap.add_argument('--pool', action='store_true', help='stylegram trained on the full curated extra pool')
    ap.add_argument('--lora_twin', type=float, default=0.0, help='use the twin-trained LoRA; add its centered evidence with this weight')
    ap.add_argument('--lora6', action='store_true', help='LoRA member = 6-adapter average (small-pool x4 + big-pool x2)')
    ap.add_argument('--lora_x', action='store_true', help='use the extra-data LoRA ensemble (OOF on folds 0-2 only)')
    args = ap.parse_args()
    fit_idx = None
    sty = 'e14_stylegram_extra_w0.5' if args.extra_w > 0 else 'e07_stylegram_svc'
    if args.pool:
        sty = 'e14_stylegram_extra_w0.5_all-ownx'
    COMPONENTS = [('stylegram', sty),
                  ('q8b_L2+22', 'k_q8b_L2+22'), ('q35_L18', 'k_q35_L18'), ('q35_L17', 'k_q35_L17'),
                  ('lora_q8b', 'lora_q8b')]
    if args.head_extra_w > 0:   # heads trained with extra data (OOF from ntl-extra-heads); q35_L17 got zero weight there
        COMPONENTS = [COMPONENTS[0], ('q8b_L2+22', 'k_q8b_L2+22_x0.3'), ('q35_L18', 'k_q35_L18_x0.3'), ('lora_q8b', 'lora_q8b')]
    heads_kernel = 'ntl-extra-heads'
    if args.heads_pool:
        COMPONENTS = [COMPONENTS[0], ('q8b_L2+22', 'kp_q8b_L2+22_x0.5'), ('q35_L18', 'kp_q35_L18_x0.5'), ('lora_q8b', 'lora_q8b')]
        heads_kernel = 'ntl-extra-heads-pool'
    if args.lora_x:
        from common import folds
        lname = 'lora_q8b_twin' if args.lora_twin > 0 else ('lora_q8b_x6' if args.lora6 else 'lora_q8b_x')
        COMPONENTS = [c for c in COMPONENTS if c[0] != 'lora_q8b'] + [(lname, 'lora_q8b_x')]
        fl = folds(load())
        fit_idx = np.concatenate([fl[f][1] for f in (0, 1, 2)])
    build(args.name, args.title, COMPONENTS, args.extra_w, args.claude_gate, args.sinkhorn, args.twin_w, args.head_extra_w,
          fit_idx, args.pool, heads_kernel, args.lora_twin)
