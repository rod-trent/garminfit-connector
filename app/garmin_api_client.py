"""
Garmin Connect API client using browser session cookies.

Replaces garth/garminconnect with direct HTTP calls to Garmin's web API.
Session cookies are obtained locally via scripts/playwright_setup.py and
stored encrypted in the database under the existing garth_token_encrypted
column (now holding a JSON payload instead of a garth base64 token).

Token JSON format:
  {"cookies": {"SESSIONID": "...", ...}, "display_name": "jsmith42"}
"""

import json
import os
import re
from datetime import date, timedelta
from typing import Optional

from curl_cffi import requests as cffi_requests

CONNECT_URL = "https://connect.garmin.com"
GC_API = f"{CONNECT_URL}/gc-api"


class GarminApiClient:
    """
    Cookie-based Garmin Connect API client.

    Implements the same method surface as garminconnect.Garmin so that
    garmin_handler.py's data methods work without modification.
    """

    def __init__(self, cookies: dict, display_name: str = "") -> None:
        # Impersonate Chrome to satisfy Cloudflare TLS fingerprint checks
        self._session = cffi_requests.Session(impersonate="chrome110")
        self._session.cookies.update(cookies)
        self._csrf: Optional[str] = None
        self.display_name = display_name

        # Route API calls through a residential proxy when set.
        # Garmin's Cloudflare blocks datacenter IPs (Railway, AWS, etc.) at the
        # API level — the same proxy used for browser login is needed here too.
        # Format: http://user:pass@host:port  or  socks5://user:pass@host:port
        _proxy = os.environ.get("RESIDENTIAL_PROXY_URL", "").strip()
        if _proxy:
            self._session.proxies = {"https": _proxy, "http": _proxy}

    # ------------------------------------------------------------------
    # Factory / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_token(cls, token_json: str) -> "GarminApiClient":
        """Deserialise from the JSON string stored in garth_token_encrypted."""
        data = json.loads(token_json)
        return cls(
            cookies=data["cookies"],
            display_name=data.get("display_name", ""),
        )

    def dumps(self) -> str:
        """Serialise for storage (replaces garth.http.Client.dumps())."""
        return json.dumps(
            {
                "cookies": dict(self._session.cookies),
                "display_name": self.display_name,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_csrf(self) -> str:
        """Lazily fetch the CSRF token from the Garmin Connect app page."""
        if self._csrf:
            return self._csrf
        resp = self._session.get(
            f"{CONNECT_URL}/modern/",
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        for pattern in [
            r'<meta[^>]+name="csrf-token"[^>]+content="([^"]+)"',
            r'<meta[^>]+content="([^"]+)"[^>]+name="csrf-token"',
            r'<meta[^>]+name="_csrf"[^>]+content="([^"]+)"',
            r'<meta[^>]+content="([^"]+)"[^>]+name="_csrf"',
        ]:
            m = re.search(pattern, resp.text, re.IGNORECASE)
            if m:
                self._csrf = m.group(1)
                return self._csrf
        raise RuntimeError(
            "Could not extract CSRF token — session may have expired. "
            "Re-run playwright_setup.py to refresh your session."
        )

    def _headers(self) -> dict:
        headers = {
            "Accept": "application/json",
            "NK": "NT",  # Garmin Connect API bypass header; sufficient without CSRF
        }
        # CSRF token is optional — the NK header covers most endpoints.
        # If extraction fails (SPA doesn't embed it in initial HTML), skip it.
        try:
            csrf = self._get_csrf()
            if csrf:
                headers["connect-csrf-token"] = csrf
        except Exception:
            pass
        return headers

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        url = f"{GC_API}{path}"
        resp = self._session.get(url, params=params, headers=self._headers())
        if resp.status_code == 401:
            self._csrf = None  # token may have rotated
            raise RuntimeError(
                "Garmin session expired (401). "
                "Re-run playwright_setup.py to get a fresh session."
            )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return resp.text

    def _post(self, path: str, json_data: Optional[dict] = None) -> object:
        url = f"{GC_API}{path}"
        headers = {**self._headers(), "Content-Type": "application/json"}
        resp = self._session.post(url, json=json_data, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _gql(self, query: str) -> dict:
        url = f"{GC_API}/graphql-gateway/graphql"
        headers = {**self._headers(), "Content-Type": "application/json"}
        resp = self._session.post(url, json={"query": query}, headers=headers)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Profile / display name
    # ------------------------------------------------------------------

    def get_full_name(self) -> Optional[str]:
        """Load display_name from Garmin social profile."""
        try:
            data = self._get("/userprofile-service/socialProfile")
            if isinstance(data, dict):
                self.display_name = (
                    data.get("displayName") or data.get("userName") or self.display_name
                )
                return data.get("fullName")
        except Exception:
            pass
        return None

    def get_user_summary(self, cdate: str) -> dict:
        return self._get(
            f"/usersummary-service/usersummary/daily/{self.display_name}",
            params={"calendarDate": cdate},
        )

    def get_stats(self, cdate: str) -> dict:
        """Alias for get_user_summary (some callers use get_stats)."""
        return self.get_user_summary(cdate)

    # ------------------------------------------------------------------
    # Activities
    # ------------------------------------------------------------------

    def get_activities(self, start: int = 0, limit: int = 20) -> list:
        data = self._get(
            "/activitylist-service/activities/search/activities",
            params={"limit": limit, "start": start},
        )
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Heart rate
    # ------------------------------------------------------------------

    def get_heart_rates(self, cdate: str) -> dict:
        return self._get(
            f"/wellness-service/wellness/dailyHeartRate/{self.display_name}",
            params={"date": cdate},
        )

    def get_rhr_day(self, cdate: str) -> dict:
        """Resting heart rate for a single day."""
        try:
            data = self._get(
                f"/usersummary-service/stats/heartRate/daily/{cdate}/{cdate}"
            )
            return data[0] if isinstance(data, list) and data else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------

    def get_sleep_data(self, cdate: str) -> dict:
        return self._get(
            f"/wellness-service/wellness/dailySleepData/{self.display_name}",
            params={"date": cdate},
        )

    # ------------------------------------------------------------------
    # Body / weight
    # ------------------------------------------------------------------

    def get_body_composition(self, cdate: str) -> dict:
        return self._get(
            "/weight-service/weight/dateRange",
            params={"startDate": cdate, "endDate": cdate},
        )

    def get_weigh_ins(self, start_date: str, end_date: str) -> dict:
        return self._get(f"/weight-service/weight/range/{start_date}/{end_date}")

    def get_daily_weigh_ins(self, cdate: str) -> dict:
        return self.get_weigh_ins(cdate, cdate)

    # ------------------------------------------------------------------
    # Stress
    # ------------------------------------------------------------------

    def get_stress_data(self, cdate: str) -> dict:
        return self._get(f"/wellness-service/wellness/dailyStress/{cdate}")

    def get_all_day_stress(self, cdate: str) -> list:
        try:
            data = self.get_stress_data(cdate)
            if isinstance(data, dict):
                return data.get("stressValuesArray", [])
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Body battery
    # ------------------------------------------------------------------

    def get_body_battery(self, cdate: str) -> dict:
        try:
            data = self._get(f"/wellness-service/wellness/bodyBattery/events/{cdate}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get_body_battery_events(self, cdate: str) -> list:
        try:
            data = self._get(f"/wellness-service/wellness/bodyBattery/events/{cdate}")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # SpO2 / respiration
    # ------------------------------------------------------------------

    def get_spo2_data(self, cdate: str) -> dict:
        return self._get(f"/wellness-service/wellness/dailySpo2/{cdate}")

    def get_respiration_data(self, cdate: str) -> dict:
        return self._get(f"/wellness-service/wellness/daily/respiration/{cdate}")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def get_steps_data(self, cdate: str) -> dict:
        try:
            data = self._get(
                f"/usersummary-service/stats/steps/daily/{cdate}/{cdate}"
            )
            return data[0] if isinstance(data, list) and data else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Hydration
    # ------------------------------------------------------------------

    def get_hydration_data(self, cdate: str) -> dict:
        return self._get(
            f"/usersummary-service/usersummary/hydration/allData/{cdate}"
        )

    # ------------------------------------------------------------------
    # HRV
    # ------------------------------------------------------------------

    def get_hrv_data(self, cdate: str) -> dict:
        try:
            data = self._get(f"/hrv-service/hrv/daily/{cdate}/{cdate}")
            return data[0] if isinstance(data, list) and data else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def get_training_status(self) -> dict:
        today = date.today().isoformat()
        try:
            return self._get(
                f"/metrics-service/metrics/trainingstatus/aggregated/{today}"
            )
        except Exception:
            return {}

    def get_training_readiness(self, cdate: str) -> dict:
        try:
            result = self._gql(
                f'query{{trainingReadinessRangeScalar(startDate:"{cdate}", endDate:"{cdate}")}}'
            )
            return result.get("data", {})
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Max metrics (VO2max etc.)
    # ------------------------------------------------------------------

    def get_max_metrics(self) -> dict:
        today = date.today()
        start = (today - timedelta(days=30)).isoformat()
        end = today.isoformat()
        try:
            return self._get(f"/metrics-service/metrics/maxmet/daily/{start}/{end}")
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def get_race_predictions(self, start_date: str, end_date: str) -> dict:
        try:
            return self._get(
                f"/metrics-service/metrics/racepredictions/daily/{self.display_name}",
                params={"fromCalendarDate": start_date, "toCalendarDate": end_date},
            )
        except Exception:
            return {}

    def get_endurance_score(self, start_date: str, end_date: str) -> dict:
        try:
            return self._get(
                "/metrics-service/metrics/endurancescore",
                params={"calendarDate": end_date},
            )
        except Exception:
            return {}

    def get_hill_score(self, start_date: str, end_date: str) -> dict:
        try:
            return self._get(
                "/metrics-service/metrics/hillscore",
                params={"calendarDate": end_date},
            )
        except Exception:
            return {}

    def get_lactate_threshold(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        try:
            return self._get("/biometric-service/lactateThreshold")
        except Exception:
            return {}

    def get_cycling_ftp(self) -> dict:
        try:
            return self._get("/biometric-service/ftpData")
        except Exception:
            return {}

    def get_running_tolerance(self, start_date: str, end_date: str) -> list:
        try:
            data = self._get(
                "/metrics-service/metrics/runningTolerance",
                params={"startDate": start_date, "endDate": end_date},
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_fitnessage_data(self, cdate: str) -> dict:
        try:
            return self._get(f"/fitnessage-service/fitnessage/{cdate}")
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Blood pressure
    # ------------------------------------------------------------------

    def get_blood_pressure(self, start_date: str, end_date: str) -> dict:
        try:
            return self._get(
                f"/bloodpressure-service/bloodpressure/daily/last/{start_date}/{end_date}"
            )
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Weekly trends
    # ------------------------------------------------------------------

    def get_weekly_steps(self, end_date: str, weeks: int = 4) -> list:
        end = date.fromisoformat(end_date)
        start = (end - timedelta(weeks=weeks)).isoformat()
        try:
            data = self._get(
                f"/usersummary-service/stats/steps/daily/{start}/{end.isoformat()}"
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_weekly_stress(self, end_date: str, weeks: int = 4) -> list:
        end = date.fromisoformat(end_date)
        start = (end - timedelta(weeks=weeks)).isoformat()
        try:
            data = self._get(
                f"/wellness-service/wellness/dailyStress/{start}/{end.isoformat()}"
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_weekly_intensity_minutes(self, start_date: str, end_date: str) -> list:
        try:
            data = self._get(
                f"/usersummary-service/stats/im/weekly/{start_date}/{end_date}"
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Nutrition
    # ------------------------------------------------------------------

    def get_nutrition_daily_food_log(self, cdate: str) -> dict:
        try:
            return self._get(
                "/usersummary-service/usersummary/nutrition-log/daily",
                params={"calendarDate": cdate},
            )
        except Exception:
            return {}

    def get_nutrition_daily_meals(self, cdate: str) -> dict:
        try:
            return self._get(
                "/usersummary-service/usersummary/nutrition-log/daily/detailed",
                params={"calendarDate": cdate},
            )
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Gear
    # ------------------------------------------------------------------

    def _get_profile_number(self) -> Optional[str]:
        try:
            data = self._get(
                "/userprofile-service/userprofile/personal-information"
            )
            if isinstance(data, dict):
                return str(data.get("profileNumber", ""))
        except Exception:
            pass
        return None

    def get_gear(self, profile_number: Optional[str] = None) -> list:
        if not profile_number:
            profile_number = self._get_profile_number()
        try:
            if profile_number:
                data = self._get(
                    "/gear-service/gear/filterGear",
                    params={"username": profile_number},
                )
            else:
                data = self._get("/gear-service/gear/v2/list")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_gear_stats(self, gear_uuid: str) -> dict:
        try:
            return self._get(f"/gear-service/gear/stats/uuid/{gear_uuid}")
        except Exception:
            return {}

    def get_gear_activities(self, gear_uuid: str, limit: int = 20) -> list:
        try:
            data = self._get(
                "/activitylist-service/activities/search/activities",
                params={"gearUUID": gear_uuid, "limit": limit},
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_activity_gear(self, activity_id: str) -> dict:
        try:
            return self._get(f"/activity-service/activity/{activity_id}/gear")
        except Exception:
            return {}

    def get_gear_defaults(self, profile_number: Optional[str] = None) -> list:
        if not profile_number:
            profile_number = self._get_profile_number()
        try:
            data = self._get(f"/gear-service/gear/defaults/{profile_number}")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Activity details
    # ------------------------------------------------------------------

    def get_activity_details(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}")

    def get_activity_splits(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}/splits")

    def get_activity_hr_in_timezones(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}/hrTimeInZones")

    def get_activity_power_in_timezones(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}/powerTimeInZones")

    def get_activity_exercise_sets(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}/exerciseSets")

    def get_activity_weather(self, activity_id: str) -> dict:
        return self._get(f"/activity-service/activity/{activity_id}/weather")

    # ------------------------------------------------------------------
    # Personal records & badges
    # ------------------------------------------------------------------

    def get_personal_record(self) -> dict:
        return self._get(
            f"/personalrecord-service/personalrecord/prs/{self.display_name}"
        )

    def get_earned_badges(self) -> list:
        try:
            data = self._get("/badge-service/badge/earned")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def get_devices(self) -> list:
        try:
            data = self._get("/device-service/deviceregistration/devices")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Garth compatibility shim
    #
    # Some code in garmin_handler.py (and mcp_server.py) accesses
    # handler.garmin.garth.dumps() or handler.garmin.garth.connectapi().
    # This shim provides just enough interface to satisfy those callers.
    # ------------------------------------------------------------------

    @property
    def garth(self) -> "_GarthShim":
        return _GarthShim(self)


class _GarthShim:
    """
    Minimal garth.http.Client-compatible shim for code that accesses
    client.garth directly (profile, connectapi, dumps).
    """

    def __init__(self, parent: "GarminApiClient") -> None:
        self._parent = parent

    def dumps(self) -> str:
        return self._parent.dumps()

    @property
    def profile(self) -> dict:
        try:
            data = self._parent._get("/userprofile-service/socialProfile")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def connectapi(self, path: str, params: Optional[dict] = None, **_) -> object:
        return self._parent._get(path, params=params)

    @property
    def display_name(self) -> str:
        return self._parent.display_name
