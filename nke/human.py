"""Human psychometric track (Sec 3.3).

Since real crowd participants cannot be recruited in this environment, this
module produces THREE deliverables:
  1. A stimulus set (PNGs) + manifest.json with ground-truth labels, condition
     (nke / gaussian-control / clean), and eps_l  -> ready for a real crowd run.
  2. A self-contained forced-choice HTML task page (the crowdsourcing harness),
     with a "cannot tell" option, that records responses to a downloadable CSV.
  3. TWO clearly-labeled *computational recognizability proxies* standing in for
     the human curve (NOT human data):
       - generic-recognizer proxy: accuracy of an independent reference model
         (different arch + seed, never exposed to the attack gradients) on the
         perturbed images. Motivated by the poor transferability of NKE examples.
       - shape proxy: accuracy of a classifier trained on Canny EDGE MAPS of
         clean images, evaluated on edge maps of perturbed images -- a direct
         computational analog of shape-biased human recognition.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from . import config as C
from . import data as D
from . import models as M


# ---------------- proxy 1: generic-recognizer (independent reference model) ----------------
@torch.no_grad()
def generic_recognizer_acc(ref_model, images, labels, batch=256):
    ref_model.eval(); correct = 0
    for i in range(0, len(images), batch):
        xb = images[i:i + batch].to(C.DEVICE)
        yb = labels[i:i + batch].to(C.DEVICE)
        correct += (ref_model(xb).argmax(1) == yb).sum().item()
    return correct / len(images)


# ---------------- proxy 2: shape / edge-map recognizer ----------------
def _edge_tensor(images, chans):
    """Batch of [0,1] images -> Canny edge maps as [N,1,H,W] float tensors."""
    outs = []
    arr = images.cpu().numpy()
    for im in arr:
        if im.shape[0] == 3:
            g = cv2.cvtColor((np.transpose(im, (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            g = (im[0] * 255).astype(np.uint8)
        e = cv2.Canny(g, 50, 150).astype(np.float32) / 255.0
        outs.append(e[None])
    return torch.from_numpy(np.stack(outs))


class EdgeNet(nn.Module):
    def __init__(self, hw, n=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * (hw // 4) * (hw // 4), 128); self.fc2 = nn.Linear(128, n)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def train_edge_recognizer(dataset, epochs=6):
    """Train a shape recognizer on Canny edge maps of clean images."""
    path = os.path.join(C.CKPT_DIR, f"{dataset}_edgenet.pt")
    hw = 28 if dataset == "mnist" else 32
    net = EdgeNet(hw, C.NUM_CLASSES[dataset]).to(C.DEVICE)
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=C.DEVICE)); net.eval(); return net
    tr = D.get_loader(dataset, train=True, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        net.train()
        for x, y in tr:
            xe = _edge_tensor(x, C.IN_CHANS[dataset]).to(C.DEVICE); y = y.to(C.DEVICE)
            opt.zero_grad(); loss = F.cross_entropy(net(xe), y); loss.backward(); opt.step()
    torch.save(net.state_dict(), path); net.eval()
    return net


@torch.no_grad()
def shape_proxy_acc(edge_net, images, labels, dataset, batch=256):
    edge_net.eval(); correct = 0
    xe = _edge_tensor(images, C.IN_CHANS[dataset])
    for i in range(0, len(xe), batch):
        xb = xe[i:i + batch].to(C.DEVICE); yb = labels[i:i + batch].to(C.DEVICE)
        correct += (edge_net(xb).argmax(1) == yb).sum().item()
    return correct / len(images)


# ---------------- stimulus + harness export ----------------
def _save_png(img, path):
    a = img.cpu().numpy()
    if a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    else:
        a = a[0]
    Image.fromarray((a * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST).save(path)


def export_stimuli_and_harness(stimset, out_dir, per_level=8):
    """Write PNG stimuli + manifest.json + index.html forced-choice task."""
    ds = stimset["dataset"]; os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images"); os.makedirs(img_dir, exist_ok=True)
    classes = C.CLASS_NAMES[ds]
    y = stimset["y"]; manifest = []
    g = np.random.RandomState(C.SEED)
    for eps, lvl in stimset["levels"].items():
        idx = g.choice(len(y), size=min(per_level, len(y)), replace=False)
        for cond, key in [("nke", "nke"), ("gauss", "gauss")]:
            if eps == 0.0 and cond == "gauss":
                continue
            for j in idx:
                fn = f"{cond}_eps{eps:g}_{int(j)}.png"
                _save_png(lvl[key][j], os.path.join(img_dir, fn))
                manifest.append({
                    "file": f"images/{fn}", "true_label": classes[int(y[j])],
                    "condition": ("clean" if eps == 0 else cond), "eps_l": float(eps),
                    "model_pred": classes[int(lvl["pred"][j])],
                    "model_conf": float(lvl["conf"][j]),
                })
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"dataset": ds, "classes": classes, "stimuli": manifest}, f, indent=2)
    _write_task_html(out_dir, ds, classes)
    print(f"[human] wrote {len(manifest)} stimuli + harness to {out_dir}")
    return len(manifest)


# Optional: set to a URL to auto-POST each participant's responses (JSON body).
# Leave empty for download-only operation (the default; no backend required).
HARNESS_POST_URL = ""

# Attention-check policy and design notes are baked into the JS below:
#   * a participant enters/receives an ID -> deterministic per-participant sampling;
#   * clean (eps_l=0) trials are shown to EVERY participant as attention checks;
#   * non-clean trials are assigned between-subjects so no participant sees the
#     same BASE image at two perturbation levels (parsed from the filename's
#     trailing index), matching the protocol in the paper.
_HARNESS_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>NKE perception study — __DATASET__</title>
<style>
body{font-family:sans-serif;text-align:center;max-width:640px;margin:2em auto}
img{image-rendering:pixelated;width:256px;height:256px;border:1px solid #ccc}
.opt{margin:4px;padding:8px 12px;font-size:15px;cursor:pointer}
#cant{background:#eee;margin-top:8px}
#prog{color:#666;margin:8px}
</style></head><body>
<h2>Which category is shown?</h2>
<p>Pick the best match. If you genuinely cannot tell, press "Cannot tell".</p>
<img id="stim" src=""><div id="prog"></div>
<div id="opts"></div>
<button id="cant" class="opt" onclick="choose('cannot_tell')">Cannot tell</button>
<script>
const CLASSES = __CLASSES_JSON__;
const POST_URL = "__POST_URL__";
let data=null, i=0, resp=[], t0=0, PID="";

// deterministic per-participant RNG (string hash -> mulberry32)
function seedFrom(s){let h=1779033703^s.length;for(let k=0;k<s.length;k++){h=Math.imul(h^s.charCodeAt(k),3432918353);h=h<<13|h>>>19;}return (h>>>0);}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
function baseIdx(file){const m=file.match(/_(\d+)\.png$/);return m?parseInt(m[1]):-1;}

function buildSession(all, rng){
  // clean trials: shown to everyone (attention checks)
  const clean = all.filter(s=>s.condition==="clean" || Number(s.eps_l)===0);
  // non-clean: one entry per base image, so no base seen at two eps levels
  const byBase={};
  all.filter(s=>!(s.condition==="clean"||Number(s.eps_l)===0))
     .forEach(s=>{const b=baseIdx(s.file);(byBase[b]=byBase[b]||[]).push(s);});
  const test=Object.values(byBase).map(g=>g[Math.floor(rng()*g.length)]);
  const sess=clean.concat(test);
  for(let k=sess.length-1;k>0;k--){const j=Math.floor(rng()*(k+1));[sess[k],sess[j]]=[sess[j],sess[k]];}
  return sess;
}
function renderOpts(){
  const o=document.getElementById('opts');o.innerHTML="";
  CLASSES.forEach(c=>{const b=document.createElement('button');b.className='opt';b.textContent=c;b.onclick=()=>choose(c);o.appendChild(b);});
}
PID = (new URLSearchParams(location.search).get('pid')) ||
      prompt("Enter your participant ID (or leave blank to auto-generate):") ||
      ("anon-"+seedFrom(""+performance.now()+Math.random()));
renderOpts();
fetch('manifest.json').then(r=>r.json()).then(m=>{
  const rng=mulberry32(seedFrom(PID));
  data=buildSession(m.stimuli, rng);
  show();
});
function show(){
  if(i>=data.length){done();return;}
  document.getElementById('stim').src=data[i].file;
  document.getElementById('prog').textContent=(i+1)+' / '+data.length+'  (ID: '+PID+')';
  t0=performance.now();
}
function choose(c){
  resp.push({...data[i], participant_id:PID, response:c, rt_ms:Math.round(performance.now()-t0)});
  i++; show();
}
function toCSV(){
  const hdr='participant_id,file,true_label,condition,eps_l,model_pred,response,rt_ms\n';
  const rows=resp.map(r=>[r.participant_id,r.file,r.true_label,r.condition,r.eps_l,r.model_pred,r.response,r.rt_ms].join(',')).join('\n');
  return hdr+rows;
}
function done(){
  const csv=toCSV();
  document.body.innerHTML='<h2>Done — thank you!</h2>';
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='responses_'+PID+'.csv';a.textContent='Download responses CSV';
  document.body.appendChild(a);
  if(POST_URL){
    fetch(POST_URL,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({participant_id:PID,dataset:"__DATASET__",responses:resp})})
      .then(()=>{const p=document.createElement('p');p.textContent='(responses submitted)';document.body.appendChild(p);})
      .catch(()=>{const p=document.createElement('p');p.textContent='(auto-submit failed — please send the downloaded CSV)';document.body.appendChild(p);});
  }
}
</script></body></html>"""


def _write_task_html(out_dir, dataset, classes):
    html = (_HARNESS_TEMPLATE
            .replace("__DATASET__", dataset)
            .replace("__CLASSES_JSON__", json.dumps(classes))
            .replace("__POST_URL__", HARNESS_POST_URL))
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)
