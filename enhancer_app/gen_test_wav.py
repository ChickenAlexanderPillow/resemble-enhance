from pathlib import Path
import math, struct, wave


def write_sine(path: Path, seconds: float = 1.0, sr: int = 48000, hz: int = 440) -> None:
    n = int(seconds * sr)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            s = int(0.2 * 32767 * math.sin(2 * math.pi * hz * i / sr))
            wf.writeframes(struct.pack('<h', s))


if __name__ == '__main__':
    out = Path('input_audio') / 'app_test.wav'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_sine(out, seconds=1.0)
    print('Wrote', out)

