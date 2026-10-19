#!/usr/bin/env python3
"""Dump per-module activation stats for one RF2 forward pass on a given device."""
import argparse, json, os, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
from hydra import compose, initialize_config_dir
import rfantibody.rf2.modules.pose_util as pu
from rfantibody.rf2.modules.preprocess import Preprocess, pose_to_inference_RFinput
from rfantibody.rf2.modules.model_runner import AbPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-hooks", type=int, default=4000)
ap.add_argument("--stop-after", default=None, help="stop when this module name is entered")
args = ap.parse_args()

cfgdir = os.path.join(ROOT, "src/rfantibody/rf2/config")
with initialize_config_dir(config_dir=cfgdir, version_base=None):
    conf = compose(config_name="base", overrides=[
        "input.pdb_dir=test/rf2/inputs_for_test",
        "output.pdb_dir=linshi/debug/rf2_layer_out",
        "inference.num_recycles=1",
        "inference.cautious=False",
        "inference.hotspot_show_proportion=0",
        "model.model_weights=weights/RF2_ab.pt",
        "+inference.seed=42",
    ])

device = torch.device(args.device)
torch.manual_seed(42)
pre = Preprocess(pose_to_inference_RFinput, conf)
pred = AbPredictor(conf, pre, device=device)
pose, tag = next(pu.pose_generator(conf))

(network_input, xyz_prev, mask_recycle, xyz_true, mask_true, msa, mask_msa, _, _, idx) = pre(pose, device=device)
network_input = pred._add_batch_dim(network_input)
input_i = {**network_input, "msa_prev": None, "pair_prev": None, "state_prev": None,
           "xyz_prev": xyz_prev[None], "mask_recycle": mask_recycle}
input_i.pop("alpha", None)
input_i["xyz_t"] = input_i["xyz_t"].clone()[..., 1, :]

capture = {}
handles = []
def mk(name):
    def hook(m, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() > 0:
            tt = t.detach().float()
            capture[name] = [float(tt.mean()), float(tt.abs().max()), list(tt.shape)]
    return hook
class _Stop(Exception):
    pass

for name, mod in pred.model.named_modules():
    if name == "" or len(handles) >= args.max_hooks:
        continue
    handles.append(mod.register_forward_hook(mk(name)))
    if args.stop_after and name == args.stop_after:
        def _stopper(m, inp, _n=name):
            raise _Stop(_n)
        handles.append(mod.register_forward_pre_hook(_stopper))

out = None
with torch.no_grad():
    try:
        out = pred.model(**input_i)
    except _Stop as e:
        print(f"stopped early at {e}")
# final output stats
final = []
for o in (out or []):
    if isinstance(o, torch.Tensor):
        oo = o.detach().float()
        final.append([float(oo.mean()), float(oo.abs().max()), list(oo.shape)])
    else:
        final.append(None)

json.dump({"device": args.device, "tag": tag, "modules": capture, "final": final},
          open(args.out, "w"))
print(f"dumped {len(capture)} module activations -> {args.out}")