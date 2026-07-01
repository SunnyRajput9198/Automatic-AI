"""
WeatherTool — real-time weather via Open-Meteo API (free, no API key needed).

Two-step:
  1. Geocode the city name → lat/lon via Open-Meteo Geocoding API
  2. Fetch current weather conditions via Open-Meteo Forecast API

Returns: temperature, feels-like, wind speed, weather description.
"""

import asyncio
import structlog
from typing import Dict, Any

import httpx

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

# WMO weather interpretation codes → human-readable description
_WMO_CODES: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
}


class WeatherTool(Tool):
    """
    Get current weather conditions for any city.
    Uses Open-Meteo (free, no API key required).
    Use for: "current temperature in X", "what is the weather in X", "temp in X".
    """

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "Get current temperature and weather conditions for any city. "
            "Free, no API key required. "
            "Use for: 'current temperature in X', 'what is the weather in X', 'temp in X'."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'New Delhi', 'Paris', 'London'",
                },
            },
            "required": ["city"],
        }

    async def run(self, **kwargs) -> ToolResult:
        city: str = kwargs.get("city", "").strip()
        if not city:
            return ToolResult(success=False, output="", error="City name is required")

        logger.info("weather_tool_running", city=city)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Step 1: geocode
                geo_resp = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": city, "count": 1, "language": "en", "format": "json"},
                )
                if geo_resp.status_code != 200:
                    return ToolResult(
                        success=False, output="",
                        error=f"Geocoding failed: HTTP {geo_resp.status_code}",
                    )

                geo_data = geo_resp.json().get("results", [])
                if not geo_data:
                    return ToolResult(
                        success=False, output="",
                        error=f"City '{city}' not found",
                    )

                lat      = geo_data[0]["latitude"]
                lon      = geo_data[0]["longitude"]
                name     = geo_data[0]["name"]
                country  = geo_data[0].get("country", "")

                # Step 2: fetch current weather
                weather_resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude":         lat,
                        "longitude":        lon,
                        "current":          "temperature_2m,apparent_temperature,weathercode,windspeed_10m,relativehumidity_2m",
                        "temperature_unit": "celsius",
                        "windspeed_unit":   "kmh",
                    },
                )
                if weather_resp.status_code != 200:
                    return ToolResult(
                        success=False, output="",
                        error=f"Weather fetch failed: HTTP {weather_resp.status_code}",
                    )

                w           = weather_resp.json().get("current", {})
                temp        = w.get("temperature_2m", "N/A")
                feels_like  = w.get("apparent_temperature", "N/A")
                wind        = w.get("windspeed_10m", "N/A")
                humidity    = w.get("relativehumidity_2m", "N/A")
                wmo_code    = w.get("weathercode", -1)
                description = _WMO_CODES.get(wmo_code, "Unknown")

            output = (
                f"Current weather in {name}, {country}:\n"
                f"  Temperature:  {temp}°C  (feels like {feels_like}°C)\n"
                f"  Conditions:   {description}\n"
                f"  Wind speed:   {wind} km/h\n"
                f"  Humidity:     {humidity}%\n"
                f"  Source: Open-Meteo (open-meteo.com)"
            )

            logger.info("weather_tool_completed", city=name, temp=temp)
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "tool_name":   "get_weather",
                    "city":        name,
                    "country":     country,
                    "temp_c":      temp,
                    "feels_like_c": feels_like,
                    "description": description,
                },
            )

        except Exception as e:
            logger.error("weather_tool_error", error=str(e), city=city)
            return ToolResult(success=False, output="", error=f"Weather lookup failed: {str(e)}")
