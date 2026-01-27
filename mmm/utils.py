from __future__ import annotations
from abc import ABC, abstractmethod
from time import perf_counter, process_time
from typing import Dict, List, Optional


# ============================================
# Base Abstract Timer
# ============================================
class BaseTimer(ABC):
    """Abstract base timer that manages named start/end intervals."""

    def __init__(self):
        self.name = self.__class__.__name__
        self._start_times: Dict[str, float] = {}
        self._durations: Dict[str, float] = {}

    @abstractmethod
    def _now(self) -> float:
        pass

    def start(self, key: str):
        self._start_times[key] = self._now()

    def stop(self, key: str):
        if key not in self._start_times:
            raise ValueError(f"Timer '{key}' was never started.")
        self._durations[key] = self._now() - self._start_times[key]

    def get(self, key: str) -> Optional[float]:
        return self._durations.get(key, None)

    def reset(self):
        self._start_times.clear()
        self._durations.clear()

    def summary(self, prefix: str = "") -> Dict[str, float]:
        """Return durations with optional prefix."""
        return {f"{prefix}{k}": v for k, v in self._durations.items()}


# ============================================
# Concrete Timers
# ============================================
class PerfCounterTimer(BaseTimer):
    def _now(self) -> float:
        return perf_counter()


class ProcessTimeTimer(BaseTimer):
    def _now(self) -> float:
        return process_time()


# ============================================
# Multi-Backend Transformer Inference Timer
# ============================================
class InferenceTimer:
    """
    Supports multiple backend timers. All timing events are broadcast
    to every backend timer automatically.
    """

    def __init__(self, timers: List[BaseTimer]):
        if not timers:
            raise ValueError("You must provide at least one timer backend.")
        self.timers = timers
        self.num_tokens = (-1, -1)

    # -------- Internal helpers --------
    def _start_all(self, key: str):
        for t in self.timers:
            t.start(key)

    def _stop_all(self, key: str):
        for t in self.timers:
            t.stop(key)

    # -------- TOKENIZATION / PREPROCESSING --------
    def start_preprocessing(self):
        self._start_all("preprocessing")

    def end_preprocessing(self):
        self._stop_all("preprocessing")

    # -------- MODEL INFERENCE (ENCODER/DECODER) ---
    def start_inference(self):
        self._start_all("inference")

    def end_inference(self):
        self._stop_all("inference")

    # -------- POST PROCESSING ---------------------
    def start_postprocessing(self):
        self._start_all("postprocessing")

    def end_postprocessing(self):
        self._stop_all("postprocessing")

    def set_num_tokens(self, input_len, gen_seq_len):
        self.num_tokens = (input_len, gen_seq_len)

    # -------- REPORT -------------------------------
    def report(self) -> Dict[str, float]:
        report = {}

        # durations per backend
        for t in self.timers:
            prefix = f"{t.name}."
            report.update(t.summary(prefix=prefix))


        report["input_length"] = self.num_tokens[0]
        report["generated_tokens_length"] = self.num_tokens[1]
        return report
