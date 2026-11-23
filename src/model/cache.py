"""In-memory TTL and LRU prediction cache for water cooler demand.

Caches predictions by (cooler_id, day_of_week, temp_band, slot_tuple) to achieve
instant response times (< 2 ms) for duplicate queries from kiosks and IoT clients,
bypassing repetitive DataFrame construction and feature transformations.
"""

from __future__ import annotations

import copy
import functools
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def compute_temp_band(outdoor_temp_c: float) -> int:
    """Map outdoor temperature in Celsius to discrete temperature band [0..4].

    Bands:
      0: < 10 °C
      1: 10 to < 18 °C
      2: 18 to < 25 °C
      3: 25 to < 32 °C
      4: >= 32 °C
    """
    if outdoor_temp_c < 10:
        return 0
    elif outdoor_temp_c < 18:
        return 1
    elif outdoor_temp_c < 25:
        return 2
    elif outdoor_temp_c < 32:
        return 3
    else:
        return 4


CacheKey = Tuple[str, str, int, Tuple[str, ...]]


class PredictionCache:
    """Thread-safe in-memory TTL/LRU keyed dictionary cache.

    Key structure: (cooler_id, day_of_week, temp_band, slot_tuple)
    """

    def __init__(self, maxsize: int = 1024, ttl_seconds: float = 1800.0):
        """Initialize prediction cache.

        Args:
            maxsize: Maximum number of entries before LRU eviction.
            ttl_seconds: Time-to-live in seconds (default 1800s / 30m). Set <= 0 to disable TTL.
        """
        self.maxsize = max(1, int(maxsize))
        self.ttl_seconds = float(ttl_seconds)
        self._cache: OrderedDict[CacheKey, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def build_key(
        cooler_id: str,
        day_of_week: str,
        temp_band: Union[int, float],
        slot_tuple: Union[Tuple[str, ...], List[str]],
    ) -> CacheKey:
        """Construct a standardized cache key tuple: (cooler_id, day_of_week, temp_band, slot_tuple)."""
        c_id = str(cooler_id).strip()
        day = str(day_of_week).capitalize()
        if isinstance(temp_band, float):
            band = compute_temp_band(temp_band)
        else:
            band = int(temp_band)
        slots = tuple(str(s) for s in slot_tuple)
        return (c_id, day, band, slots)

    def get(self, key: CacheKey) -> Optional[Any]:
        """Retrieve entry from cache. Returns None if missing or expired."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            created_at, value = self._cache[key]
            if self.ttl_seconds > 0 and (time.time() - created_at) > self.ttl_seconds:
                del self._cache[key]
                self.misses += 1
                return None

            # Mark as recently used (LRU order)
            self._cache.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(value)

    def set(self, key: CacheKey, value: Any) -> None:
        """Store an entry in the cache with the current timestamp."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), copy.deepcopy(value))
            if len(self._cache) > self.maxsize:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def __contains__(self, key: CacheKey) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            created_at, _ = self._cache[key]
            if self.ttl_seconds > 0 and (time.time() - created_at) > self.ttl_seconds:
                del self._cache[key]
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def currsize(self) -> int:
        return len(self)

    def cache_info(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "maxsize": self.maxsize,
                "currsize": len(self._cache),
                "ttl_seconds": self.ttl_seconds,
            }


def make_lru_cached_predictor(
    predict_fn: Callable[[str, str, int, Tuple[str, ...]], Any],
    maxsize: int = 1024,
) -> Callable[[str, str, int, Tuple[str, ...]], Any]:
    """Wrap a prediction function with standard functools.lru_cache for (cooler_id, day_of_week, temp_band, slot_tuple)."""
    @functools.lru_cache(maxsize=maxsize)
    def _lru_wrapped(
        cooler_id: str,
        day_of_week: str,
        temp_band: int,
        slot_tuple: Tuple[str, ...],
    ) -> Any:
        return predict_fn(cooler_id, day_of_week, temp_band, slot_tuple)

    return _lru_wrapped
