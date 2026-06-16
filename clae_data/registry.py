from __future__ import annotations

from clae_data.adapters import (
    CommonVoiceBnAdapter,
    IndicVoicesAdapter,
    KathbathAdapter,
    OpenSLR53Adapter,
    RegSpeech12Adapter,
    ShrutilipiAdapter,
    SubakKoAdapter,
)
from clae_data.adapters.base import DatasetAdapter

REGISTRY: dict[str, type[DatasetAdapter]] = {
    "openslr53": OpenSLR53Adapter,
    "common_voice_bn": CommonVoiceBnAdapter,
    "regspeech12": RegSpeech12Adapter,
    "indicvoices": IndicVoicesAdapter,
    "subak_ko": SubakKoAdapter,
    "shrutilipi": ShrutilipiAdapter,
    "kathbath": KathbathAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()
