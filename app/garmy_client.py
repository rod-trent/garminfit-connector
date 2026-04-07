"""
Garmy-based Garmin Connect API client using OAuth2 tokens.

Replaces GarminApiClient (cookie-based) for users who authenticate via the
server-side email/password flow.  Authentication is handled by garmy's
AuthClient; data requests go through garmy's APIClient.connectapi(), which
routes to https://connectapi.garmin.com/{path} using an OAuth2 Bearer token —
the same endpoint paths we already use, just a different subdomain.

Token JSON format (stored encrypted in garth_token_encrypted):
  {
    "type": "garmy_oauth",
    "oauth1": {
      "oauth_token": "...",
      "oauth_token_secret": "...",
      "mfa_token": null,
      "mfa_expiration_timestamp": null,
      "domain": "garmin.com"
    },
    "oauth2": {
      "scope": "...",
      "jti": "...",
      "token_type": "Bearer",
      "access_token": "...",
      "refresh_token": "...",
      "expires_in": 3600,
      "expires_at": 1234567890,
      "refresh_token_expires_in": 7776000,
      "refresh_token_expires_at": 1234567890
    },
    "display_name": "jsmith42"
  }
"""

import json
from datetime import date, datetime, timedelta
from typing import Optional

from garmy import AuthClient, APIClient
from garmy.auth.tokens import OAuth1Token, OAuth2Token

TOKEN_TYPE = "garmy_oauth"


def is_garmy_token(token_json: str) -> bool:
    """Return True if *token_json* is a garmy OAuth token (not a cookie token)."""
    try:
        data = json.loads(token_json)
        return data.get("type") == TOKEN_TYPE
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Garth compatibility shim
# ---------------------------------------------------------------------------

class _GarmyGarthShim:
    """
    Minimal garth.http.Client-compatible shim so that code paths that call
    handler.garmin.garth.dumps() continue to work unchanged.
    """

    def __init__(self, parent: "GarmyApiClient") -> None:
        self._parent = parent

    def dumps(self) -> str:
        return self._parent.dumps()

    @property
    def display_name(self) -> str:
        return self._parent.display_name

    def connectapi(self, path: str, params: Optional[dict] = None, **_) -> object:
        return self._parent._get(path, params=params)


# ---------------------------------------------------------------------------
# GarmyApiClient
# ---------------------------------------------------------------------------

class GarmyApiClient:
    """
    OAuth2-based Garmin Connect API client backed by garmy.

    Exposes the same method interface as GarminApiClient so that
    MultiUserGarminHandler (and garmin_handler.py's data methods) work
    without any changes.
    """

    def __init__(
        self,
        auth_client: AuthClient,
        api_client: APIClient,
        display_name: str = "",
    ) -> None:
        self._auth = auth_client
        self._api = api_client
        self.display_name = display_name

    # ------------------------------------------------------------------
    # Factory / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_token(cls, token_json: str) -> "GarmyApiClient":
        """Restore a GarmyApiClient from the JSON string stored in the DB."""
        data = json.loads(token_json)

        # Build OAuth1Token — mfa_expiration_timestamp may be an ISO string
        oauth1_data = dict(data["oauth1"])
        ts = oauth1_data.get("mfa_expiration_timestamp")
        if isinstance(ts, str):
            oauth1_data["mfa_expiration_timestamp"] = datetime.fromisoformat(ts)

        oauth1_token = OAuth1Token(**oauth1_data)
        oauth2_token = OAuth2Token(**data["oauth2"])

        # Create AuthClient without loading tokens from disk, then inject ours
        auth_client = AuthClient()
        auth_client.token_manager.set_tokens(oauth1_token, oauth2_token)

        api_client = APIClient(auth_client=auth_client)

        return cls(auth_client, api_client, data.get("display_name", ""))

    def dumps(self) -> str:
        """Serialise the current OAuth tokens + display_name to JSON for DB storage."""
        oauth1: OAuth1Token = self._auth.token_manager.oauth1_token
        oauth2: OAuth2Token = self._auth.token_manager.oauth2_token

        ts = oauth1.mfa_expiration_timestamp
        oauth1_dict = {
            "oauth_token": oauth1.oauth_token,
            "oauth_token_secret": oauth1.oauth_token_secret,
            "mfa_token": oauth1.mfa_token,
            "mfa_expiration_timestamp": ts.isoformat() if ts else None,
            "domain": oauth1.domain,
        }

        oauth2_dict = {
            "scope": oauth2.scope,
            "jti": oauth2.jti,
            "token_type": oauth2.token_type,
            "access_token": oauth2.access_token,
            "refresh_token": oauth2.refresh_token,
            "expires_in": oauth2.expires_in,
            "expires_at": oauth2.expires_at,
            "refresh_token_expires_in": oauth2.refresh_token_expires_in,
            "refresh_token_expires_at": oauth2.refresh_token_expires_at,
        }

        return json.dumps({
            "type": TOKEN_TYPE,
            "oauth1": oauth1_dict,
            "oauth2": oauth2_dict,
            "display_name": self.display_name,
        })

    # ------------------------------------------------------------------
    # Garth compatibility shim (used by save_refreshed_tokens)
    # ------------------------------------------------------------------

    @property
    def garth(self) -> "_GarmyGarthShim":
        return _GarmyGarthShim(self)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> object:
        """GET via garmy's connectapi (connectapi.garmin.com)."""
        kwargs = {}
        if params:
            kwargs["params"] = params
        return self._api.connectapi(path, **kwargs)

    def _gql(self, query: str) -> dict:
        """GraphQL via garmy."""
        return self._api.graphql(query)

    # ------------------------------------------------------------------
    # Profile / display name
    # ------------------------------------------------------------------

    def get_full_name(self) -> Optional[str]:
        """Load display_name from the Garmin social profile."""
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
