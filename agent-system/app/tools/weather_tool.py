"""Current weather with a retried Open-Meteo primary and wttr.in fallback."""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx
import structlog

from app.tools.base import Tool, ToolResult, classify_tool_failure

logger = structlog.get_logger()
_WMO_CODES: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Light showers",
    81: "Showers", 82: "Violent showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
}


class WeatherTool(Tool):
    """Retrieve current city conditions from Open-Meteo, falling back to wttr.in."""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Retrieve current observed weather for a city. Primary source is Open-Meteo; "
            "after transient/provider failure it retries and may use wttr.in as a separately "
            "attributed fallback source. Returns temperature, feels-like, wind, humidity, "
            "and conditions. Required input: city. This provides current conditions, not historical weather."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. New Delhi, Paris, London"}
        }, "required": ["city"]}

    @staticmethod
    def _failure(error: str, status_code: Optional[int] = None) -> ToolResult:
        return ToolResult(success=False, output="", error=error, metadata={
            "tool_name": "get_weather",
            "failure_type": classify_tool_failure(error, status_code),
            "source": "Open-Meteo",
        })

    async def _get_retrying(self, client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
        """Retry once for a transient network failure or provider 5xx only."""
        for attempt in range(2):
            try:
                response = await client.get(url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt:
                    raise
                logger.warning("weather_provider_retry", failure_type="NETWORK_OR_TIMEOUT")
                await asyncio.sleep(0.4)
                continue
            if response.status_code >= 500 and attempt == 0:
                logger.warning("weather_provider_retry", status_code=response.status_code)
                await asyncio.sleep(0.4)
                continue
            return response
        raise httpx.RequestError("Weather provider request failed")

    async def _fallback_wttr(self, city: str, primary_error: str) -> ToolResult:
        """Try an independent weather source and retain its provenance."""
        try:
            url = "https://wttr.in/" + quote(city, safe="")
            async with httpx.AsyncClient(timeout=12, follow_redirects=False,
                                         headers={"User-Agent": "AgentSystem/1.0"}) as client:
                response = await client.get(url, params={"format": "j1"})
            if response.status_code != 200:
                failure = self._failure(
                    f"Weather providers unavailable (Open-Meteo: {primary_error}; wttr.in HTTP {response.status_code})",
                    response.status_code,
                )
                failure.metadata["provider_attempts"] = [
                    {"source": "Open-Meteo", "status": "failed", "failure_type": classify_tool_failure(primary_error)},
                    {"source": "wttr.in", "status": "failed", "failure_type": classify_tool_failure(None, response.status_code)},
                ]
                return failure
            current = response.json().get("current_condition", [])
            area = response.json().get("nearest_area", [])
            if not current:
                return self._failure("wttr.in returned no current conditions")
            weather = current[0]
            area_name = area[0].get("areaName", [{}])[0].get("value", city) if area else city
            country = area[0].get("country", [{}])[0].get("value", "") if area else ""
            condition = weather.get("weatherDesc", [{}])[0].get("value", "Unknown")
            temp = weather.get("temp_C", "N/A")
            feels = weather.get("FeelsLikeC", "N/A")
            wind = weather.get("windspeedKmph", "N/A")
            humidity = weather.get("humidity", "N/A")
            output = (
                f"Current weather in {area_name}, {country}:\n"
                f"  Temperature: {temp}°C (feels like {feels}°C)\n"
                f"  Conditions: {condition}\n  Wind speed: {wind} km/h\n"
                f"  Humidity: {humidity}%\n  Source: wttr.in (fallback weather provider)"
            )
            return ToolResult(success=True, output=output, metadata={
                "tool_name": self.name, "source": "wttr.in", "fallback_from": "Open-Meteo",
                "primary_failure_type": classify_tool_failure(primary_error),
                "provider_attempts": [
                    {"source": "Open-Meteo", "status": "failed", "failure_type": classify_tool_failure(primary_error)},
                    {"source": "wttr.in", "status": "success"},
                ],
                "city": area_name, "country": country, "temp_c": temp,
                "feels_like_c": feels, "description": condition,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("weather_fallback_failed", error_type=type(exc).__name__)
            failure = self._failure(
                f"Weather data unavailable from Open-Meteo and wttr.in ({type(exc).__name__})"
            )
            failure.metadata["provider_attempts"] = [
                {"source": "Open-Meteo", "status": "failed", "failure_type": classify_tool_failure(primary_error)},
                {"source": "wttr.in", "status": "failed", "failure_type": classify_tool_failure(type(exc).__name__)},
            ]
            return failure

    async def run(self, **kwargs: Any) -> ToolResult:
        city = str(kwargs.get("city", "")).strip()
        if not city:
            return self._failure("City name is required")
        logger.info("weather_tool_running", city=city)
        primary_error = "unknown failure"
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                geo_resp = await self._get_retrying(
                    client, "https://geocoding-api.open-meteo.com/v1/search",
                    {"name": city, "count": 1, "language": "en", "format": "json"},
                )
                if geo_resp.status_code != 200:
                    primary_error = f"geocoding HTTP {geo_resp.status_code}"
                    if geo_resp.status_code < 500 and geo_resp.status_code not in (408, 429):
                        return self._failure(primary_error, geo_resp.status_code)
                    return await self._fallback_wttr(city, primary_error)
                locations = geo_resp.json().get("results", [])
                if not locations:
                    return await self._fallback_wttr(city, "city not found by geocoder")
                location = locations[0]
                weather_resp = await self._get_retrying(
                    client, "https://api.open-meteo.com/v1/forecast",
                    {"latitude": location["latitude"], "longitude": location["longitude"],
                     "current": "temperature_2m,apparent_temperature,weathercode,windspeed_10m,relativehumidity_2m",
                     "temperature_unit": "celsius", "windspeed_unit": "kmh"},
                )
                if weather_resp.status_code != 200:
                    primary_error = f"forecast HTTP {weather_resp.status_code}"
                    if weather_resp.status_code < 500 and weather_resp.status_code not in (408, 429):
                        return self._failure(primary_error, weather_resp.status_code)
                    return await self._fallback_wttr(city, primary_error)
                current = weather_resp.json().get("current", {})
                if not current:
                    return await self._fallback_wttr(city, "Open-Meteo returned no current conditions")
            name = location.get("name", city)
            country = location.get("country", "")
            code = current.get("weathercode", -1)
            condition = _WMO_CODES.get(int(code), "Unknown")
            temp = current.get("temperature_2m", "N/A")
            feels = current.get("apparent_temperature", "N/A")
            wind = current.get("windspeed_10m", "N/A")
            humidity = current.get("relativehumidity_2m", "N/A")
            output = (
                f"Current weather in {name}, {country}:\n"
                f"  Temperature: {temp}°C (feels like {feels}°C)\n"
                f"  Conditions: {_WMO_CODES.get(int(code), 'Unknown')}\n"
                f"  Wind speed: {wind} km/h\n  Humidity: {humidity}%\n"
                "  Source: Open-Meteo (open-meteo.com)"
            )
            return ToolResult(success=True, output=output, metadata={
                "tool_name": self.name, "source": "Open-Meteo", "city": name,
                "country": country, "temp_c": temp, "feels_like_c": feels,
                "description": condition, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            primary_error = type(exc).__name__
            logger.warning("weather_provider_failed", failure_type=classify_tool_failure(primary_error))
            return await self._fallback_wttr(city, primary_error)
