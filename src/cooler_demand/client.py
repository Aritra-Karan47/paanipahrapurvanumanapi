"""HTTP client for the Hugging Face Gradio / inference service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schemas import (
    ForecastDayRequest,
    ForecastDayResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    ShortageCheckRequest,
    ShortageCheckResponse,
    TemperatureTargetRequest,
    TemperatureTargetResponse,
    ValidateDataRequest,
    ValidateDataResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3


class CoolerDemandError(Exception):
    """Base exception for SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class CoolerDemandClient:
    """
    Client for the School Water Demand Prediction Hugging Face Space.

    The main application owns institute config, DB, sensors, and weather API.
    This client only talks to the stateless prediction service.
    """

    def __init__(
        self,
        base_url: str,
        api_token: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ):
        """
        Args:
            base_url: Hugging Face Space URL, e.g.
                      https://username-school-water-demand.hf.space
                      or http://localhost:7860 for local Gradio.
            api_token: Optional HF token if the Space is gated.
            timeout: Request timeout seconds.
            max_retries: Transient failure retries.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CoolerDemandClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _url(self, path: str) -> str:
        # Gradio API style or custom REST endpoints exposed by the Space
        if path.startswith("/"):
            return path
        return f"/{path}"

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(DEFAULT_RETRIES),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._url(path)
        logger.debug("POST %s", url)
        try:
            resp = self._client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise CoolerDemandError(f"Request timed out after {self.timeout}s", detail=str(exc)) from exc
        except httpx.TransportError as exc:
            raise CoolerDemandError(f"Transport error: {exc}") from exc

        if resp.status_code >= 400:
            detail = None
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise CoolerDemandError(
                f"Service error {resp.status_code}",
                status_code=resp.status_code,
                detail=detail,
            )
        return resp.json()

    def _call_gradio(self, api_name: str, data: List[Any]) -> Any:
        """Call a Gradio endpoint via /call/{api_name} (Gradio 4+)."""
        # First enqueue
        enqueue = self._post(f"/call/{api_name}", {"data": data})
        event_id = enqueue.get("event_id")
        if not event_id:
            # Some Spaces return result directly
            return enqueue
        # Poll result
        with self._client.stream("GET", f"/call/{api_name}/{event_id}") as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    import json
                    payload = json.loads(line[5:].strip())
                    if isinstance(payload, list) and payload:
                        return payload[0] if len(payload) == 1 else payload
                    return payload
        raise CoolerDemandError("No result from Gradio event stream")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, request: Union[PredictionRequest, Dict[str, Any]]) -> PredictionResponse:
        """Predict water demand for one or more intervals of a cooler."""
        if isinstance(request, dict):
            request = PredictionRequest.model_validate(request)
        payload = request.model_dump(mode="json")
        try:
            raw = self._post("/api/predict", payload)
        except CoolerDemandError:
            # Fallback to Gradio named endpoint if custom REST not present
            raw = self._call_gradio("predict", [payload])
        return PredictionResponse.model_validate(raw)

    def forecast_day(self, request: Union[ForecastDayRequest, Dict[str, Any]]) -> ForecastDayResponse:
        """Forecast a full day of intervals and compute shortage risk."""
        if isinstance(request, dict):
            request = ForecastDayRequest.model_validate(request)
        payload = request.model_dump(mode="json")
        try:
            raw = self._post("/api/forecast_day", payload)
        except CoolerDemandError:
            raw = self._call_gradio("forecast_day", [payload])
        return ForecastDayResponse.model_validate(raw)

    def get_temperature_target(
        self, request: Union[TemperatureTargetRequest, float, Dict[str, Any]]
    ) -> TemperatureTargetResponse:
        """Deterministic outdoor → target water temperature mapping."""
        if isinstance(request, (int, float)):
            request = TemperatureTargetRequest(outdoor_temp_c=float(request))
        elif isinstance(request, dict):
            request = TemperatureTargetRequest.model_validate(request)
        payload = request.model_dump(mode="json")
        try:
            raw = self._post("/api/temperature_target", payload)
        except CoolerDemandError:
            raw = self._call_gradio("temperature_target", [payload])
        return TemperatureTargetResponse.model_validate(raw)

    def check_shortage(
        self, request: Union[ShortageCheckRequest, Dict[str, Any]]
    ) -> ShortageCheckResponse:
        """Evaluate shortage risk given capacity, level, and predicted demand."""
        if isinstance(request, dict):
            request = ShortageCheckRequest.model_validate(request)
        payload = request.model_dump(mode="json")
        try:
            raw = self._post("/api/check_shortage", payload)
        except CoolerDemandError:
            raw = self._call_gradio("check_shortage", [payload])
        return ShortageCheckResponse.model_validate(raw)

    def validate_data(
        self, request: Union[ValidateDataRequest, Dict[str, Any]]
    ) -> ValidateDataResponse:
        """Validate cooler / timetable / consumption payloads before prediction."""
        if isinstance(request, dict):
            request = ValidateDataRequest.model_validate(request)
        payload = request.model_dump(mode="json")
        try:
            raw = self._post("/api/validate_data", payload)
        except CoolerDemandError:
            raw = self._call_gradio("validate_data", [payload])
        return ValidateDataResponse.model_validate(raw)

    def model_info(self) -> ModelInfo:
        """Return service model version and capability flags."""
        try:
            raw = self._client.get(self._url("/api/model_info")).json()
        except Exception:
            raw = self._call_gradio("model_info", [])
        return ModelInfo.model_validate(raw)