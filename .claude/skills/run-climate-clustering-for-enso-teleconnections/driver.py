#!/usr/bin/env python3
"""Programmatic driver for the climate-clustering-for-ENSO-teleconnections repo.

Every subcommand chdir()s to the repo root first, because the configs and the
dataset loader use paths relative to it (`../data/era5/*.nc`, `checkpoint/...`).

Run it with the project venv:
    .venv/bin/python .claude/skills/run-climate-clustering-for-enso-teleconnections/driver.py <cmd>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_CONFIG = 'checkpoint/temperature-era-amazonia-15/params-temperature-era-amazonia-15.yaml'
DEFAULT_RESULTS = 'results/temperature-era-amazonia-15'
SHOT_DIR = REPO / '.claude' / 'skills' / 'run-climate-clustering-for-enso-teleconnections' / 'shots'


def ok(msg):
    print(f'  \033[32mOK\033[0m   {msg}')


def bad(msg):
    print(f'  \033[31mFAIL\033[0m {msg}')


# --------------------------------------------------------------------------- #
# env: is this machine able to run anything at all?
# --------------------------------------------------------------------------- #
def cmd_env(args):
    import torch
    print(f'repo         : {REPO}')
    print(f'python       : {sys.executable}')
    print(f'torch        : {torch.__version__}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu          : {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')

    rc = 0
    era5 = sorted(glob.glob('../data/era5/*.nc'))
    temp = [f for f in era5 if 'min-max-temp' in f]
    prep = [f for f in era5 if 'total-prep' in f]
    (ok if temp else bad)(f'ERA5 temperature files: {len(temp)}')
    (ok if prep else bad)(f'ERA5 precipitation files: {len(prep)}')
    rc |= 0 if temp else 1

    oni = Path('../data/oni_index.xlsx')
    (ok if oni.exists() else bad)(f'ONI index: {oni}')
    rc |= 0 if oni.exists() else 1

    cache = sorted(glob.glob('../data/era5/.daily_cache/*.pt'))
    print(f'  daily caches : {len(cache)} '
          f'({", ".join(Path(c).name for c in cache) if cache else "none - first load will take ~40 min"})')

    for d in ('checkpoint', 'results', 'configs', 'logs'):
        Path(d).mkdir(exist_ok=True)
    ck = sorted(glob.glob('checkpoint/*/params-*.yaml'))
    print(f'  runs with a dumped config: {len(ck)}')
    for c in ck:
        print(f'    {c}')
    return rc


# --------------------------------------------------------------------------- #
# data: exercise the dataset layer only (no model). Fast when the cache is warm.
# --------------------------------------------------------------------------- #
def cmd_data(args):
    import yaml
    import torch
    from src.data_manager import Era5Dataset

    p = yaml.safe_load(open(args.config))['data']
    t0 = time.time()
    ds = Era5Dataset(transform=None, surf_vars=p['surf_vars'],
                     lat_lim=p['lat_limit'], lon_lim=p['lon_limit'],
                     split_val=True, val_years=tuple(p['val_years']),
                     deseason_standardize=p.get('deseason_standardize', False))
    dt = time.time() - t0
    print(f'surf_vars     : {p["surf_vars"]}')
    print(f'train tensor  : {tuple(ds.imgs.shape)}  ({ds.time[0].date()} -> {ds.time[-1].date()})')
    if ds.validation_imgs is not None:
        print(f'val tensor    : {tuple(ds.validation_imgs.shape)}  '
              f'({ds.val_time[0].date()} -> {ds.val_time[-1].date()})')
    print(f'load time     : {dt:.1f}s   (>60s means the daily cache was cold)')
    for i, v in enumerate(p['surf_vars']):
        ch = ds.imgs[:, i]
        print(f'  {v:>5}: mean={ch.mean():+.3f} std={ch.std():.3f} '
              f'min={ch.min():+.2f} max={ch.max():+.2f} nan={int(torch.isnan(ch).sum())}')
    assert not torch.isnan(ds.imgs).any(), 'NaNs in the training tensor'
    ok('dataset layer')
    return 0


# --------------------------------------------------------------------------- #
# infer: load a checkpoint and push real days through the encoder.
# This is the direct-invocation path - use it for PRs that touch src/.
# --------------------------------------------------------------------------- #
def cmd_infer(args):
    import torch
    import numpy as np
    import src.inference as inf

    params, dataset = inf.read_data(args.config, validation=False)
    encoder, prot = inf.load_model(params, model_suffix=args.suffix)
    n = args.n
    # dataset[i] applies the eval transform (Resize+CenterCrop+Normalize).
    # Feeding dataset.imgs directly is native resolution and blows up in the
    # positional embedding: "size of tensor a (36) must match tensor b (26)".
    with torch.no_grad():
        batch = torch.stack([dataset[i][0] for i in range(n)]).to('cuda:0')
        emb = torch.nn.functional.normalize(encoder(batch))
        logits = emb @ prot.T
    print(f'config        : {args.config}')
    print(f'checkpoint    : {params["logging"]["write_tag"]}{args.suffix}')
    print(f'input batch   : {tuple(batch.shape)}')
    print(f'embedding     : {tuple(emb.shape)}   prototypes: {tuple(prot.shape)}')
    print(f'cluster ids   : {logits.argmax(1).cpu().numpy().tolist()}')
    print(f'max softmax   : {logits.softmax(1).max(1).values.mean():.3f} (mean over batch)')
    assert emb.shape[0] == n and prot.shape[0] == params['criterion']['num_proto']
    ok('encoder + prototypes')
    return 0


# --------------------------------------------------------------------------- #
# clusters: the analysis layer, from cached F tensors (no GPU needed).
# --------------------------------------------------------------------------- #
def cmd_clusters(args):
    import yaml
    import torch
    import numpy as np
    import pandas as pd
    from src.data_manager import Era5Dataset
    from src.el_nino import read_enso_data
    import src.enso_stats as es

    p = yaml.safe_load(open(args.config))
    f = Path(args.results) / 'F.pt'
    fv = Path(args.results) / 'F_val.pt'
    if not f.exists():
        bad(f'{f} missing - run the analysis notebook once to populate it')
        return 1
    d = p['data']
    ds = Era5Dataset(transform=None, surf_vars=d['surf_vars'],
                     lat_lim=d['lat_limit'], lon_lim=d['lon_limit'],
                     split_val=True, val_years=tuple(d['val_years']),
                     deseason_standardize=d.get('deseason_standardize', False))
    F = torch.load(f, weights_only=False)
    N = p['criterion']['num_proto']
    cid = F.argmax(1).cpu().numpy()
    dates = list(ds.time)
    if fv.exists():
        Fv = torch.load(fv, weights_only=False)
        cid = np.concatenate([cid, Fv.argmax(1).cpu().numpy()])
        dates += list(ds.val_time)
    df = pd.DataFrame({'cluster_id': cid, 'date': pd.to_datetime(dates)})
    oni = read_enso_data(); oni.index.name = 'date_period'
    monthly = es.build_monthly(df, oni, N)
    stat, pv = es.omnibus_test(monthly, N, n_perm=args.perm)
    print(f'days          : {len(df)}   clusters occupied: {len(np.unique(cid))}/{N}')
    print(f'months        : {len(monthly)}   ENSO episodes: {monthly.episode.nunique()}')
    print(f'cluster freq %: {np.round(np.bincount(cid, minlength=N)/len(cid)*100, 1).tolist()}')
    print(f'ENSO omnibus  : stat={stat:.3f}  p={pv:.4f}  '
          f'({"signal" if pv < 0.05 else "no global signal"})')
    ok('analysis layer')
    return 0


# --------------------------------------------------------------------------- #
# train-smoke: a real 2-epoch training run into a scratch checkpoint dir.
# --------------------------------------------------------------------------- #
def cmd_train_smoke(args):
    import yaml
    base = yaml.safe_load(open(args.config))
    tag = 'smoke-run'
    folder = f'checkpoint/{tag}/'
    base['optimization']['epochs'] = args.epochs
    base['optimization']['warmup'] = 1
    base['optimization']['val_patience'] = 0
    base['logging'] = {'folder': folder, 'write_tag': tag}
    Path(folder).mkdir(parents=True, exist_ok=True)
    cfg = f'configs/_smoke.yaml'
    yaml.safe_dump(base, open(cfg, 'w'))
    print(f'smoke config  : {cfg}  ({args.epochs} epochs -> {folder})')

    log = Path('logs/smoke_train.log')
    log.parent.mkdir(exist_ok=True)
    t0 = time.time()
    with open(log, 'w') as lf:
        r = subprocess.run([sys.executable, 'main_msn.py', '--fname', cfg,
                            '--devices', args.device],
                           stdout=lf, stderr=subprocess.STDOUT)
    txt = log.read_text()
    losses = [l for l in txt.splitlines() if 'avg. loss' in l]
    print(f'exit={r.returncode}  elapsed={time.time()-t0:.0f}s  epochs logged={len(losses)}')
    for l in losses:
        print('   ', l.strip())
    if r.returncode != 0 or len(losses) < args.epochs:
        bad(f'training smoke failed - see {log}')
        print('\n'.join(txt.splitlines()[-25:]))
        return 1
    ckpts = sorted(glob.glob(f'{folder}*.pth.tar'))
    print(f'checkpoints   : {[Path(c).name for c in ckpts]}')
    ok('training pipeline')
    return 0


# --------------------------------------------------------------------------- #
# app / shot: the Streamlit viewer.
# --------------------------------------------------------------------------- #
def _wait_port(port, timeout=180):
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(1)
    return False


def cmd_app(args):
    log = Path('logs/streamlit.log')
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    proc = subprocess.Popen(
        [str(Path(sys.executable).parent / 'streamlit'), 'run', 'streamlit_app.py',
         '--server.headless', 'true', '--server.port', str(args.port),
         '--server.address', '127.0.0.1'],
        stdout=open(log, 'w'), stderr=subprocess.STDOUT, env=env)
    if not _wait_port(args.port):
        bad(f'streamlit did not open port {args.port}; see {log}')
        proc.kill()
        return 1
    print(f'streamlit pid={proc.pid} on http://127.0.0.1:{args.port}  (log: {log})')
    print('stop it with: pkill -f "streamlit run streamlit_app.py"')
    if args.wait:
        proc.wait()
    return 0


def cmd_shot(args):
    """Drive the Streamlit UI with Playwright: fill the sidebar with a config that
    actually exists, wait for the tabs to render, screenshot each one."""
    from playwright.sync_api import sync_playwright

    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    url = f'http://127.0.0.1:{args.port}'
    out = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True, args=['--no-sandbox', '--disable-gpu'])
        pg = b.new_page(viewport={'width': 1600, 'height': 1000})
        pg.goto(url, wait_until='domcontentloaded', timeout=60_000)
        pg.wait_for_selector('section[data-testid="stSidebar"]', timeout=60_000)

        # The shipped defaults point at checkpoint/temperature-run2/... which does
        # not exist in this repo, so the app errors until these are replaced.
        for label, value in (('Config file', args.config), ('Results path', args.results)):
            box = pg.get_by_label(label)
            box.fill(value)
            box.press('Enter')
            pg.wait_for_timeout(1500)

        # Streamlit keeps a "Running..." status widget up while reruns are in flight.
        for _ in range(args.timeout):
            if pg.locator('[data-testid="stStatusWidget"]').count() == 0:
                break
            pg.wait_for_timeout(1000)
        pg.wait_for_selector('button[role="tab"]', timeout=120_000)

        err = pg.locator('[data-testid="stException"]').count()
        p0 = SHOT_DIR / 'app-loaded.png'
        pg.screenshot(path=str(p0), full_page=False)
        out.append(p0)

        tabs = pg.locator('button[role="tab"]')
        names = [tabs.nth(i).inner_text().strip() for i in range(tabs.count())]
        print(f'tabs: {names}')
        for i, nm in enumerate(names):
            if args.tabs and nm not in args.tabs:
                continue
            tabs.nth(i).click()
            pg.wait_for_timeout(args.settle)
            pth = SHOT_DIR / f'tab-{i}-{nm.lower().replace(" ", "-")}.png'
            pg.screenshot(path=str(pth), full_page=False)
            out.append(pth)
        b.close()

    for p in out:
        print(f'  wrote {p}  ({p.stat().st_size//1024} KB)')
    if err:
        bad(f'{err} Streamlit exception block(s) rendered - check app-loaded.png')
        return 1
    ok('streamlit UI rendered')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('env', help='check venv, GPU, data files, caches').set_defaults(fn=cmd_env)

    p = sub.add_parser('data', help='load the dataset only (no model)')
    p.add_argument('--config', default=DEFAULT_CONFIG); p.set_defaults(fn=cmd_data)

    p = sub.add_parser('infer', help='load a checkpoint and encode N real days')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--suffix', default='-latest.pth.tar')
    p.add_argument('--n', type=int, default=8); p.set_defaults(fn=cmd_infer)

    p = sub.add_parser('clusters', help='ENSO stats from cached F tensors')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--results', default=DEFAULT_RESULTS)
    p.add_argument('--perm', type=int, default=2000); p.set_defaults(fn=cmd_clusters)

    p = sub.add_parser('train-smoke', help='real 2-epoch training run')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--device', default='cuda:0'); p.set_defaults(fn=cmd_train_smoke)

    p = sub.add_parser('app', help='launch the Streamlit viewer')
    p.add_argument('--port', type=int, default=8501)
    p.add_argument('--wait', action='store_true'); p.set_defaults(fn=cmd_app)

    p = sub.add_parser('shot', help='drive the running Streamlit app and screenshot it')
    p.add_argument('--port', type=int, default=8501)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--results', default=DEFAULT_RESULTS)
    p.add_argument('--tabs', nargs='*', default=None)
    p.add_argument('--settle', type=int, default=3000)
    p.add_argument('--timeout', type=int, default=300); p.set_defaults(fn=cmd_shot)

    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == '__main__':
    main()
