from __future__ import annotations

from clae_data.adapters.base import DatasetAdapter
from clae_data.adapters.bengaliai_speech import BengaliAISpeechAdapter
from clae_data.adapters.indicvoices import IndicVoicesAdapter
from clae_data.adapters.kathbath import KathbathAdapter
from clae_data.adapters.openslr53 import OpenSLR53Adapter
from clae_data.adapters.regspeech12 import RegSpeech12Adapter
from clae_data.adapters.shrutilipi import ShrutilipiAdapter
from clae_data.adapters.subak_ko import SubakKoAdapter

__all__ = [
    "DatasetAdapter",
    "BengaliAISpeechAdapter",
    "IndicVoicesAdapter",
    "KathbathAdapter",
    "OpenSLR53Adapter",
    "RegSpeech12Adapter",
    "ShrutilipiAdapter",
    "SubakKoAdapter",
]
