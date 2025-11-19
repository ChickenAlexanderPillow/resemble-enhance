import math
from pathlib import Path
import shutil
import torch
import torchaudio
import audalign as ad
base = Path('tmp_audalign_test2')
if base.exists():
    shutil.rmtree(base, ignore_errors=True)
base.mkdir(exist_ok=True)
sr = 48000
secs = 6
samples = sr * secs
x = torch.linspace(0, secs, steps=samples)
w1 = torch.sin(2 * math.pi * 220 * x) * 0.3
w2 = torch.zeros_like(w1)
shift = int(0.25 * sr)
w2[shift:] = w1[:-shift]
f1 = base / 'track1.wav'
f2 = base / 'track2.wav'
torchaudio.save(str(f1), w1.unsqueeze(0), sr)
torchaudio.save(str(f2), w2.unsqueeze(0), sr)
res = ad.align_files(str(f1), str(f2))
print('RESULT KEYS', res.keys())
print(res)
