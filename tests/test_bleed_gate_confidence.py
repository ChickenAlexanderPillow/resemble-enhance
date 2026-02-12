import math
import os
import io
import re
import unittest
from contextlib import redirect_stdout

import torch

from enhancer_gui import _apply_bleed_gate


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x) + 1e-12)


def _attenuation_db(before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> float:
    b = before[mask]
    a = after[mask]
    if b.numel() == 0:
        return 0.0
    ratio = float(_rms(a) / (_rms(b) + 1e-12))
    return max(0.0, -20.0 * math.log10(max(1e-9, ratio)))


def _carrier(sr: int, dur_s: float) -> torch.Tensor:
    n = int(sr * dur_s)
    t = torch.arange(n, dtype=torch.float32) / float(sr)
    return (0.65 * torch.sin(2 * math.pi * 180.0 * t) + 0.35 * torch.sin(2 * math.pi * 260.0 * t))


def _mask(sr: int, a: float, b: float, total: int) -> torch.Tensor:
    m = torch.zeros(total, dtype=torch.bool)
    i0 = max(0, int(a * sr))
    i1 = min(total, int(b * sr))
    if i1 > i0:
        m[i0:i1] = True
    return m


class BleedGateConfidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.sr = 16000
        self._prev_hard_iso = os.environ.get("RESEMBLE_BLEED_HARD_ISOLATION")
        os.environ["RESEMBLE_BLEED_HARD_ISOLATION"] = "0"

    def tearDown(self) -> None:
        if self._prev_hard_iso is None:
            os.environ.pop("RESEMBLE_BLEED_HARD_ISOLATION", None)
        else:
            os.environ["RESEMBLE_BLEED_HARD_ISOLATION"] = self._prev_hard_iso

    def test_single_speaker_alternation_two_channels(self) -> None:
        dur = 6.0
        n = int(self.sr * dur)
        sig = _carrier(self.sr, dur)
        a_env = (_mask(self.sr, 0.0, 2.0, n) | _mask(self.sr, 4.0, 6.0, n)).float()
        b_env = _mask(self.sr, 2.0, 4.0, n).float()
        a = sig * a_env
        b = sig * b_env
        bleed = 0.28
        ch1 = a + bleed * b
        ch2 = b + bleed * a
        out1, out2 = _apply_bleed_gate([ch1, ch2], self.sr)

        m_a = _mask(self.sr, 0.0, 2.0, n) | _mask(self.sr, 4.0, 6.0, n)
        m_b = _mask(self.sr, 2.0, 4.0, n)
        self.assertGreater(_attenuation_db(ch2, out2, m_a), 8.0)
        self.assertGreater(_attenuation_db(ch1, out1, m_b), 8.0)
        self.assertLess(_attenuation_db(ch1, out1, m_a), 3.5)
        self.assertLess(_attenuation_db(ch2, out2, m_b), 3.5)

    def test_double_talk_overlap_preserved(self) -> None:
        dur = 5.0
        n = int(self.sr * dur)
        sig = _carrier(self.sr, dur)
        a_env = (_mask(self.sr, 0.0, 1.2, n) | _mask(self.sr, 1.2, 3.8, n)).float()
        b_env = (_mask(self.sr, 1.2, 3.8, n) | _mask(self.sr, 3.8, 5.0, n)).float()
        a = sig * a_env
        b = sig * b_env
        bleed = 0.22
        ch1 = a + bleed * b
        ch2 = b + bleed * a
        out1, out2 = _apply_bleed_gate([ch1, ch2], self.sr)

        overlap = _mask(self.sr, 1.2, 3.8, n)
        self.assertLess(_attenuation_db(ch1, out1, overlap), 3.0)
        self.assertLess(_attenuation_db(ch2, out2, overlap), 3.0)

    def test_near_equal_ambiguous_stays_open(self) -> None:
        dur = 4.0
        n = int(self.sr * dur)
        sig = _carrier(self.sr, dur)
        noise_a = 0.01 * torch.randn(n)
        noise_b = 0.01 * torch.randn(n)
        ch1 = sig + noise_a
        ch2 = sig + noise_b
        out1, out2 = _apply_bleed_gate([ch1, ch2], self.sr)
        full = torch.ones(n, dtype=torch.bool)
        self.assertLess(_attenuation_db(ch1, out1, full), 2.0)
        self.assertLess(_attenuation_db(ch2, out2, full), 2.0)

    def test_rapid_turn_taking_avoids_false_cuts(self) -> None:
        dur = 3.0
        n = int(self.sr * dur)
        sig = _carrier(self.sr, dur)
        seg = int(0.12 * self.sr)
        a_env = torch.zeros(n)
        b_env = torch.zeros(n)
        for i in range(0, n, seg):
            j = min(n, i + seg)
            if (i // seg) % 2 == 0:
                a_env[i:j] = 1.0
            else:
                b_env[i:j] = 1.0
        a = sig * a_env
        b = sig * b_env
        bleed = 0.25
        ch1 = a + bleed * b
        ch2 = b + bleed * a
        out1, out2 = _apply_bleed_gate([ch1, ch2], self.sr)

        active_a = a_env > 0
        active_b = b_env > 0
        self.assertLess(_attenuation_db(ch1, out1, active_a), 4.0)
        self.assertLess(_attenuation_db(ch2, out2, active_b), 4.0)

    def test_n_channel_sanity_four_channels(self) -> None:
        dur = 4.0
        n = int(self.sr * dur)
        sig = _carrier(self.sr, dur)
        clean = [torch.zeros(n) for _ in range(4)]
        for i in range(4):
            m = _mask(self.sr, float(i), float(i + 1), n).float()
            clean[i] = sig * m
        bleed = 0.2
        chans = []
        for i in range(4):
            x = clean[i].clone()
            for j in range(4):
                if i != j:
                    x = x + bleed * clean[j]
            chans.append(x)
        out = _apply_bleed_gate(chans, self.sr)
        self.assertEqual(len(out), 4)
        for i in range(4):
            self.assertEqual(int(out[i].numel()), n)
            self.assertTrue(torch.isfinite(out[i]).all().item())

        m0 = _mask(self.sr, 0.0, 1.0, n)
        non_winner_atts = [_attenuation_db(chans[i], out[i], m0) for i in range(1, 4)]
        self.assertGreater(sum(non_winner_atts) / len(non_winner_atts), 6.0)

    def test_stable_dominance_limits_chatter(self) -> None:
        dur = 8.0
        n = int(self.sr * dur)
        t = torch.arange(n, dtype=torch.float32) / float(self.sr)
        base = 0.22 * torch.sin(2 * math.pi * 220.0 * t)
        # Speech-like burst envelope with pauses for realistic noise-floor estimation.
        env = torch.zeros(n)
        on = int(0.28 * self.sr)
        off = int(0.10 * self.sr)
        i = 0
        while i < n:
            j = min(n, i + on)
            env[i:j] = 1.0
            i = j + off
        # Channel 1 is consistently stronger; channel 2 tracks it with lower gain.
        ch1 = (base * env) + 0.003 * torch.randn(n)
        ch2 = (0.48 * base * env) + 0.004 * torch.randn(n)
        out1, out2 = _apply_bleed_gate([ch1, ch2], self.sr)

        full = env > 0
        # Winner should remain mostly intact while loser gets meaningful suppression.
        self.assertLess(_attenuation_db(ch1, out1, full), 2.5)
        self.assertGreater(_attenuation_db(ch2, out2, full), 2.0)

    def test_hard_isolation_mode_strong_loser_suppression(self) -> None:
        prev_iso = os.environ.get("RESEMBLE_BLEED_HARD_ISOLATION")
        prev_att = os.environ.get("RESEMBLE_BLEED_HARD_ATT_DB")
        prev_min_conf = os.environ.get("RESEMBLE_BLEED_MIN_ATT_CONF")
        try:
            dur = 4.0
            n = int(self.sr * dur)
            sig = _carrier(self.sr, dur)
            a_env = (_mask(self.sr, 0.0, 2.0, n) | _mask(self.sr, 3.0, 4.0, n)).float()
            b_env = _mask(self.sr, 2.0, 3.0, n).float()
            a = sig * a_env
            b = sig * b_env
            bleed = 0.35
            ch1 = a + bleed * b
            ch2 = b + bleed * a
            os.environ["RESEMBLE_BLEED_MIN_ATT_CONF"] = "0"
            os.environ["RESEMBLE_BLEED_HARD_ISOLATION"] = "0"
            cap0 = io.StringIO()
            with redirect_stdout(cap0):
                _apply_bleed_gate([ch1, ch2], self.sr)
            os.environ["RESEMBLE_BLEED_HARD_ISOLATION"] = "1"
            os.environ["RESEMBLE_BLEED_HARD_ATT_DB"] = "42"
            cap1 = io.StringIO()
            with redirect_stdout(cap1):
                _apply_bleed_gate([ch1, ch2], self.sr)

            def _extract_mean_att(s: str) -> float:
                m = re.search(r"mean_loser_att_db=([0-9]+(?:\.[0-9]+)?)", s)
                if not m:
                    return 0.0
                return float(m.group(1))

            base_mean = _extract_mean_att(cap0.getvalue())
            hard_mean = _extract_mean_att(cap1.getvalue())
            self.assertGreater(hard_mean, base_mean + 10.0)
        finally:
            if prev_iso is None:
                os.environ.pop("RESEMBLE_BLEED_HARD_ISOLATION", None)
            else:
                os.environ["RESEMBLE_BLEED_HARD_ISOLATION"] = prev_iso
            if prev_att is None:
                os.environ.pop("RESEMBLE_BLEED_HARD_ATT_DB", None)
            else:
                os.environ["RESEMBLE_BLEED_HARD_ATT_DB"] = prev_att
            if prev_min_conf is None:
                os.environ.pop("RESEMBLE_BLEED_MIN_ATT_CONF", None)
            else:
                os.environ["RESEMBLE_BLEED_MIN_ATT_CONF"] = prev_min_conf


if __name__ == "__main__":
    unittest.main()
