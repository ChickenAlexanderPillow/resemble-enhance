from dataclasses import dataclass

from ..hparams import HParams as HParamsBase


@dataclass(frozen=True)
class HParams(HParamsBase):
    batch_size_per_gpu: int = 128
    distort_prob: float = 0.5
    denoiser_l1_weight: float = 0.8
    denoiser_mrstft_sc_weight: float = 0.7
    denoiser_mrstft_mag_weight: float = 0.9
    denoiser_sisdr_weight: float = 0.25
    denoiser_sisdr_target_db: float = 20.0
    denoiser_vad_margin_db: float = 8.0
    denoiser_vad_speech_boost: float = 2.0
