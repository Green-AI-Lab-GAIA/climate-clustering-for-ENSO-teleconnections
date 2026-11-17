"""Does the inference crop (Resize(146)+CenterCrop(128): only 61% of the domain) matter?

Re-encode every day with the FULL field squeezed to 128x128 (Resize((128,128))), using
the same -best checkpoint, then compare assignments and ENSO statistics.
usage: crop_test.py <scratch_dir> <review_dir> <run label> [<run label> ...]
"""
import sys, os, re
sys.path.insert(0, sys.argv[1])
from review_lib import *
import torchvision.transforms as T
import src.inference as inf

oni = oni_table()
for label in sys.argv[3:]:
    rdir, cfg = RUNS[label]
    params = yaml.safe_load(open(cfg))
    p = params['data']
    ds = Era5Dataset(transform=None, surf_vars=p['surf_vars'], deseason_standardize=p['deseason_standardize'],
                     var_transform=p.get('var_transform'), **DATA_KW)
    enc, prot = inf.load_model(params, model_suffix='-best.pth.tar')
    tf = T.Compose([T.Resize((128, 128)), T.Normalize(p['norm_means'], p['norm_stds'])])

    def encode(imgs):
        out = []
        with torch.no_grad():
            for s in range(0, len(imgs), 256):
                x = torch.stack([tf(im) for im in imgs[s:s + 256]]).cuda()
                z = torch.nn.functional.normalize(enc(x))
                out.append((z @ prot.T).cpu())
        return torch.cat(out)

    Ff, Ffv = encode(ds.imgs), encode(ds.validation_imgs)
    slug = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
    od = os.path.join(sys.argv[2], f'fullfield_{slug}'); os.makedirs(od, exist_ok=True)
    torch.save(Ff, f'{od}/F.pt'); torch.save(Ffv, f'{od}/F_val.pt')

    ctr, cv, k, *_ = run_assignments(rdir)
    ftr, fv = Ff.argmax(1).numpy(), Ffv.argmax(1).numpy()
    agree = np.r_[ftr == ctr, fv == cv].mean()
    t_tr, t_v = pd.DatetimeIndex(ds.time), pd.DatetimeIndex(ds.val_time)
    for nm, a, b in (('center crop (current)', ctr, cv), ('full field', ftr, fv)):
        m = monthly_from(a, b, t_tr, t_v, k, oni)
        r = exact_rotation_stats(m, k, min_shift=24)
        print(f'{label:<12} {nm:<22} omnibus S={r["stat"]:.3f} p={r["p_omni"]:.4f} | nFWER={int((r["p_fwer"]<0.05).sum())} '
              f'| max|r|={np.abs(r["corr"]).max():.3f}')
    print(f'{label:<12} same cluster for {agree:.1%} of days; cluster sizes full-field '
          f'{np.bincount(ftr, minlength=k).min()/len(ftr)*100:.1f}-{np.bincount(ftr, minlength=k).max()/len(ftr)*100:.1f}%\n')
