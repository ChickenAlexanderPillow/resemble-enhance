from dataclasses import dataclass

from ..hparams import HParams as HParamsBase


@dataclass(frozen=True)
class HParams(HParamsBase):
    batch_size_per_gpu: int = 128
    distort_prob: float = 0.5
    denoiser_l1_weight: float = 1.0
    denoiser_mrstft_sc_weight: float = 0.5
    denoiser_mrstft_mag_weight: float = 0.5
    denoiser_sisdr_weight: float = 0.25
    denoiser_sisdr_target_db: float = 20.0
