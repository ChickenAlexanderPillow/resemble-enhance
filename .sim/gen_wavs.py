import torch
import torchaudio
from pathlib import Path
sr = 48000
T = 3.0
N = int(T*sr)
t = torch.linspace(0, T, N)
# Two tones with same content but shifted
x = 0.25*torch.sin(2*3.14159*440*t) + 0.05*torch.sin(2*3.14159*880*t)
# Add a short chirp start
x[:2000] += torch.hann_window(2000)*0.5
shift_s = 0.35
shift = int(shift_s*sr)
y = torch.zeros_like(x)
y[shift:shift+N-shift] = x[:N-shift]
base = Path('output_audio/test_gui_sync')
base.mkdir(parents=True, exist_ok=True)
A = base/'A.wav'
B = base/'B.wav'
torchaudio.save(str(A), x.unsqueeze(0), sr)
torchaudio.save(str(B), y.unsqueeze(0), sr)
print('WAVS', A, B)
