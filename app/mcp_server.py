"""
FastMCP server with 16 Garmin data tools.

All tools:
  - Read the user's access token from a contextvars.ContextVar set by GarminMCPRouter
  - Run blocking garminconnect calls in a thread executor (never block the event loop)
  - Default to today's date when no date is provided
  - Return formatted strings suitable for Claude to read and summarize

Tool groupings:
  Group 1 — Daily Overview:     get_health_snapshot, get_today_summary, get_sleep_summary, get_activities
  Group 2 — Recovery & Wellness: get_body_battery, get_stress_summary, get_hrv_status, get_heart_rate
  Group 3 — Training Performance: get_training_status, get_training_readiness, get_intensity_minutes
  Group 4 — Nutrition & Hydration: get_nutrition_log, get_hydration
  Group 5 — Comprehensive & Range: get_body_metrics, get_spo2_and_respiration, get_activities_by_date_range
"""

import asyncio
import contextvars
from datetime import datetime, timedelta
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.garmin_adapter import get_garmin_handler, save_refreshed_tokens

# ---------------------------------------------------------------------------
# ContextVar — set by GarminMCPRouter for each request
# ---------------------------------------------------------------------------

# This variable holds the current user's access token during a request.
# It is set in app/main.py's GarminMCPRouter before delegating to the MCP app.
user_access_token_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "user_access_token"
)


def _get_token() -> str:
    """Retrieve the current user's access token from the context variable."""
    try:
        return user_access_token_var.get()
    except LookupError:
        raise RuntimeError(
            "No user access token found in request context. "
            "Make sure you are connecting via your personal MCP URL."
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _yesterday() -> str:
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Core helper: get handler + run blocking call in thread pool
# ---------------------------------------------------------------------------

async def _call(method_name: str, *args, **kwargs) -> str:
    """
    Get an authenticated MultiUserGarminHandler for the current user,
    run the named method in a thread executor, return the result as a string.
    """
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    def _run():
        method = getattr(handler, method_name)
        return method(*args, **kwargs)

    result = await loop.run_in_executor(None, _run)

    # Persist any token refresh that occurred
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if result is None:
        return "No data available for this metric."
    return str(result) if not isinstance(result, str) else result


async def _format(data_type: str, activity_limit: int = 5) -> str:
    """Call format_data_for_context() which wraps multiple Garmin sub-calls."""
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None, handler.format_data_for_context, data_type, activity_limit
    )

    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if not result:
        return "No data available for this metric."
    return str(result)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "Garmin Fitness",
    json_response=True,     # Plain JSON responses, no SSE streaming
    stateless_http=True,    # Fresh server per request — no session state needed
    instructions=(
        "Tools for querying the connected user's Garmin Connect fitness data. "
        "All date parameters use YYYY-MM-DD format. "
        "When no date is specified, today's date is used automatically. "
        "Data availability depends on the user's Garmin device model — "
        "some metrics (e.g. nutrition, hydration, body composition) require "
        "manual logging or specific Garmin hardware. "
        "Start with get_health_snapshot for a comprehensive daily overview."
    ),
)


# ===========================================================================
# GROUP 1 — Daily Overview
# ===========================================================================

@mcp.tool()
async def get_health_snapshot(date: Optional[str] = None) -> str:
    """
    Returns a comprehensive health data snapshot for one day.
    Includes steps, calories, sleep, body battery, stress, HRV,
    heart rate, and training status — all in one response.

    This is the best tool for a complete daily health overview.
    Use this instead of calling multiple individual tools.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Formatted text summary of all available health metrics for that day.
    """
    if not date:
        date = _today()
    return await _format("comprehensive")


@mcp.tool()
async def get_today_summary(date: Optional[str] = None) -> str:
    """
    Returns a daily activity summary: step count, calories burned,
    distance, active minutes, and goal progress.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Formatted summary of daily activity totals.
    """
    if not date:
        date = _today()
    return await _format("summary")


@mcp.tool()
async def get_sleep_summary(date: Optional[str] = None) -> str:
    """
    Returns last night's sleep data: total sleep duration, time in each
    sleep stage (deep, light, REM, awake), and sleep score if available.

    Note: Sleep data is recorded for the PREVIOUS night. Pass yesterday's
    date (or no date) to get the most recent sleep data.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to yesterday if not provided
              (since sleep is recorded for the prior night).

    Returns:
        Formatted sleep breakdown in hours and minutes.
    """
    if not date:
        date = _yesterday()
    return await _format("sleep")


@mcp.tool()
async def get_activities(
    limit: Optional[int] = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Returns recent workout and activity records: activity type (running,
    cycling, swimming, etc.), duration, distance, pace, calories, and
    heart rate zones for each activity.

    Args:
        limit: Number of activities to return (1-20). Defaults to 5.
        start_date: Optional start date filter in YYYY-MM-DD format.
        end_date: Optional end date filter in YYYY-MM-DD format.

    Returns:
        List of recent activities with key stats for each.
    """
    limit = max(1, min(20, limit or 5))

    if start_date and end_date:
        return await _call("get_activities_by_date", start_date, end_date)

    return await _format("activities", limit)


# ===========================================================================
# GROUP 2 — Recovery & Wellness
# ===========================================================================

@mcp.tool()
async def get_body_battery(date: Optional[str] = None) -> str:
    """
    Returns Garmin Body Battery energy level data: current level (0-100),
    daily high and low, and charge/drain events throughout the day.

    Body Battery reflects overall energy reserves combining sleep quality,
    stress, and activity. High values indicate good recovery.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Body Battery levels and charge/drain events for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_body_battery, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No Body Battery data available for this date."
    return await _format("body_battery")


@mcp.tool()
async def get_stress_summary(date: Optional[str] = None) -> str:
    """
    Returns stress level data: average stress score, max stress,
    and time spent in each stress category (low, medium, high, rest).

    Garmin stress is measured via heart rate variability (0-100 scale).
    Lower scores indicate less physiological stress.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Stress scores and time breakdown across stress categories.
    """
    if not date:
        date = _today()
    return await _format("stress")


@mcp.tool()
async def get_hrv_status(date: Optional[str] = None) -> str:
    """
    Returns Heart Rate Variability (HRV) data: last night's HRV average,
    5-day baseline, and HRV status (balanced, unbalanced, poor).

    HRV is a key recovery indicator — higher values generally indicate
    better nervous system recovery. Best viewed over time as a trend.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        HRV measurements and status classification.
    """
    if not date:
        date = _today()
    return await _format("hrv")


@mcp.tool()
async def get_heart_rate(date: Optional[str] = None) -> str:
    """
    Returns heart rate data for the day: resting heart rate,
    maximum heart rate recorded, and average heart rate.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Resting, average, and max heart rate for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_heart_rate_data, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No heart rate data available for this date."
    import json
    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# GROUP 3 — Training Performance
# ===========================================================================

@mcp.tool()
async def get_training_status() -> str:
    """
    Returns training performance metrics: VO2 Max estimate, fitness age,
    training load (acute vs. chronic), and current training status
    (productive, maintaining, recovery, overreaching, detraining).

    These metrics update after activities and reflect fitness trends
    over the past 4 weeks.

    Returns:
        VO2 Max, fitness age, training load, and training status classification.
    """
    return await _format("training")


@mcp.tool()
async def get_training_readiness(date: Optional[str] = None) -> str:
    """
    Returns today's training readiness score (0-100) and the contributing
    factors: sleep quality, HRV status, recovery time, body battery,
    and recent training load.

    Higher scores indicate your body is well-recovered and ready for
    hard training. Below 50 suggests a lighter day is advisable.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Training readiness score and factor breakdown.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_training_readiness, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No training readiness data available. This metric requires a compatible Garmin device."
    import json
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_intensity_minutes(date: Optional[str] = None) -> str:
    """
    Returns weekly moderate and vigorous intensity activity minutes,
    compared to the WHO-recommended 150 minutes of moderate activity per week.

    Vigorous activity counts double toward the weekly goal.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Weekly intensity minutes breakdown and goal progress percentage.
    """
    if not date:
        date = _today()
    return await _format("intensity")


# ===========================================================================
# GROUP 4 — Nutrition & Hydration
# ===========================================================================

@mcp.tool()
async def get_nutrition_log(date: Optional[str] = None) -> str:
    """
    Returns nutrition data logged in Garmin Connect: total calories consumed,
    macronutrients (protein, carbs, fat, fiber, sugar), and individual
    meal/food entries if logged.

    Note: This data is only populated if the user manually logs food in the
    Garmin Connect app or a connected food tracking service.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Calories consumed, macros breakdown, and food log entries.
    """
    if not date:
        date = _today()
    return await _format("nutrition")


@mcp.tool()
async def get_hydration(date: Optional[str] = None) -> str:
    """
    Returns water intake logged in Garmin Connect for the day,
    in both milliliters and US cups.

    Note: This data is only populated if the user logs water intake
    in the Garmin Connect app.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Total water intake in ml and cups for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_hydration_data, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No hydration data logged for this date. Log water intake in the Garmin Connect app."
    import json
    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# GROUP 5 — Comprehensive & Date-Range
# ===========================================================================

@mcp.tool()
async def get_body_metrics(date: Optional[str] = None) -> str:
    """
    Returns body composition metrics from a compatible Garmin scale:
    weight, BMI, body fat percentage, muscle mass, and bone mass.

    Note: Requires a Garmin Index smart scale or manual body composition
    entry in Garmin Connect.

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        Body composition measurements for the given date.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, handler.get_body_composition, date)
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))
    if not result:
        return "No body composition data for this date. Requires a Garmin Index scale."
    import json
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_spo2_and_respiration(date: Optional[str] = None) -> str:
    """
    Returns blood oxygen saturation (SpO2) and breathing/respiration rate data.

    SpO2 measures the percentage of oxygen in the blood — normal is 95-100%.
    Respiration rate is breaths per minute, which is often lower during sleep.

    Note: Requires a Garmin device with pulse oximeter (e.g. Fenix, Forerunner 945+).

    Args:
        date: Date in YYYY-MM-DD format. Defaults to today if not provided.

    Returns:
        SpO2 percentage and respiration rate data for the day.
    """
    if not date:
        date = _today()
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()

    spo2, resp = await asyncio.gather(
        loop.run_in_executor(None, handler.get_spo2_data, date),
        loop.run_in_executor(None, handler.get_respiration_data, date),
    )

    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    import json
    return json.dumps(
        {"spo2": spo2, "respiration": resp},
        indent=2, default=str
    )


@mcp.tool()
async def get_activities_by_date_range(start_date: str, end_date: str) -> str:
    """
    Returns all workout activities between two dates — useful for weekly
    or monthly summaries. Each activity includes type, duration, distance,
    pace, calories, and average heart rate.

    Args:
        start_date: Start of date range in YYYY-MM-DD format (e.g. "2026-03-01").
        end_date: End of date range in YYYY-MM-DD format (e.g. "2026-03-13").

    Returns:
        All activities within the specified date range.
    """
    token = _get_token()
    handler = await get_garmin_handler(token)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, handler.get_activities_by_date, start_date, end_date
    )
    asyncio.create_task(save_refreshed_tokens(token, handler.garmin))

    if not result:
        return f"No activities found between {start_date} and {end_date}."
    import json
    return json.dumps(result, indent=2, default=str)
