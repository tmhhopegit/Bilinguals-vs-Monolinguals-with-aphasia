"""Do lesion-based outcome models trained on monolingual (L1) stroke patients
transfer to bilingual patients assessed in their second language (L2)?

Python port of the MATLAB Bilinguals project.
"""
from .config import Config
from .data import load_patients, task_data

__all__ = ["Config", "load_patients", "task_data"]
