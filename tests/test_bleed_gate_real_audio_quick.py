from pathlib import Path
import unittest

import torch
try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

from enhancer_gui import _apply_bleed_gate


class BleedGateRealAudioQuickTests(unittest.TestCase):
    def test_quick_real_audio_smoke(self) -> None:
        if torchaudio is None:
            self.skipTest("torchaudio is not available in this environment")
        repo = Path(__file__).resolve().parents[1]
        p1 = repo / "test" / "00004_Interviewer.WAV"
        p2 = repo / "test" / "00021_Guest.WAV"
        if not (p1.exists() and p2.exists()):
            self.skipTest("Expected test audio files are missing in ./test")

        w1, sr1 = torchaudio.load(str(p1))
        w2, sr2 = torchaudio.load(str(p2))
        self.assertGreater(int(sr1), 0)
        self.assertGreater(int(sr2), 0)

        if int(sr1) != int(sr2):
            w2 = torchaudio.functional.resample(w2, int(sr2), int(sr1))
        sr = int(sr1)

        mono1 = (w1.mean(dim=0) if w1.dim() == 2 else w1).float()
        mono2 = (w2.mean(dim=0) if w2.dim() == 2 else w2).float()
        n_total = min(int(mono1.numel()), int(mono2.numel()))
        n = min(n_total, int(sr * 1.5))
        self.assertGreater(n, int(sr * 0.25))

        starts = [0]
        mid = max(0, (n_total // 2) - (n // 2))
        if mid + n <= n_total and mid > 0:
            starts.append(mid)

        for start in starts:
            with self.subTest(start=start):
                ch1 = mono1[start:start + n]
                ch2 = mono2[start:start + n]
                out = _apply_bleed_gate([ch1, ch2], sr)
                self.assertEqual(len(out), 2)
                self.assertEqual(int(out[0].numel()), n)
                self.assertEqual(int(out[1].numel()), n)
                self.assertTrue(torch.isfinite(out[0]).all().item())
                self.assertTrue(torch.isfinite(out[1]).all().item())

                # Smoke-level behavior check: algorithm should be non-expansive in energy.
                in_rms = torch.sqrt(torch.mean(ch1 * ch1) + torch.mean(ch2 * ch2) + 1e-12)
                out_rms = torch.sqrt(torch.mean(out[0] * out[0]) + torch.mean(out[1] * out[1]) + 1e-12)
                self.assertLessEqual(float(out_rms), float(in_rms) * 1.01)


if __name__ == "__main__":
    unittest.main()
