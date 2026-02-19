import torch

from enhancer_gui import _build_speaker_turns, _is_low_info_text, _select_camera_for_speaker


def test_low_info_text_filtering():
    assert _is_low_info_text("hmm mhmm")
    assert _is_low_info_text("yeah")
    assert not _is_low_info_text("yeah that sounds good")
    assert not _is_low_info_text("we should keep this segment")


def test_build_speaker_turns_filters_and_merges():
    # 100ms hop, 20 frames total.
    winner = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    conf = torch.tensor([0.8] * 20, dtype=torch.float32)
    turns = _build_speaker_turns(
        winner_idx=winner,
        conf=conf,
        hop_ms=100.0,
        n_speakers=2,
        min_conf=0.66,
        min_dur_s=0.4,
        merge_gap_s=0.25,
    )
    assert len(turns) == 3
    assert turns[0]["speaker_idx"] == 0
    assert turns[1]["speaker_idx"] == 1
    assert turns[2]["speaker_idx"] == 0
    assert turns[0]["end_s"] == 0.5
    assert turns[1]["end_s"] == 1.0


def test_select_camera_priority_exact_closeup_first():
    roles = {
        "wide": "wide.mov",
        "guest_closeup": "guest.mov",
        "host_closeup": "host.mov",
        "extra_guest_1": "guest_alt.mov",
    }
    assert _select_camera_for_speaker(0, host_idx=0, wide="wide.mov", camera_roles=roles) == "host.mov"
    assert _select_camera_for_speaker(1, host_idx=0, wide="wide.mov", camera_roles=roles) == "guest.mov"
    roles2 = {"wide": "wide.mov"}
    assert _select_camera_for_speaker(1, host_idx=0, wide="wide.mov", camera_roles=roles2) == "wide.mov"

