"""Tests for stable contract DTOs and serialization."""

import unittest

from fall_prediction_service.contracts import (
    HealthResponse,
    MonitorCommandResponse,
    PaginatedResponse,
    PredictionDTO,
    ProfileDTO,
    ServiceErrorDTO,
    SettingsDTO,
    StatusResponse,
)
from fall_prediction_service.errors import (
    ServiceError,
    unauthorized,
    service_not_ready,
    camera_in_use,
    monitor_already_running,
    import_conflict,
    invalid_argument,
    model_load_failed,
    database_error,
    not_found,
)
from fall_prediction_service.serialization import (
    clamp01,
    serialize_event,
    serialize_health,
    serialize_monitor_command,
    serialize_paginated,
    serialize_profile,
    serialize_settings,
    serialize_status,
)


class TestClamp(unittest.TestCase):
    def test_in_range(self):
        self.assertEqual(clamp01(0.5), 0.5)

    def test_below_zero(self):
        self.assertEqual(clamp01(-0.5), 0.0)

    def test_above_one(self):
        self.assertEqual(clamp01(1.5), 1.0)

    def test_zero(self):
        self.assertEqual(clamp01(0.0), 0.0)

    def test_one(self):
        self.assertEqual(clamp01(1.0), 1.0)


class TestSerializeHealth(unittest.TestCase):
    def test_starting(self):
        result = serialize_health(status="starting")
        self.assertEqual(result["status"], "starting")
        self.assertEqual(result["api_version"], "v1")
        self.assertIn("models", result)

    def test_ready(self):
        result = serialize_health(
            status="ready",
            models_loaded=True,
            database_ok=True,
            camera_available=True,
        )
        self.assertEqual(result["status"], "ready")
        self.assertTrue(result["database"])
        self.assertTrue(result["camera_available"])

    def test_degraded(self):
        result = serialize_health(status="degraded", database_ok=False)
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["database"])


class TestSerializeStatus(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "running": True,
            "loading": False,
            "state": "Normal",
            "riskPercent": 18,
            "confidencePercent": 93,
            "fps": 24.6,
            "error": "",
            "systemStatus": None,
        }

    def test_normal_state(self):
        result = serialize_status(self.snapshot)
        self.assertEqual(result["schema_version"], 1)
        self.assertTrue(result["monitoring"])
        self.assertFalse(result["loading"])
        pred = result["prediction"]
        self.assertEqual(pred["state"], "Normal")
        self.assertEqual(pred["business_state"], "safe")
        self.assertAlmostEqual(pred["risk_score"], 0.18)
        self.assertAlmostEqual(pred["confidence"], 0.93)

    def test_fall_state(self):
        snap = {**self.snapshot, "state": "High Risk Detected", "riskPercent": 85}
        result = serialize_status(snap)
        self.assertEqual(result["prediction"]["state"], "Fall")
        self.assertEqual(result["prediction"]["business_state"], "danger")

    def test_confirmed_fall_never_serializes_with_low_display_risk(self):
        snap = {**self.snapshot, "state": "Fall", "riskPercent": 14}
        result = serialize_status(snap)
        self.assertEqual(result["prediction"]["business_state"], "danger")
        self.assertAlmostEqual(result["prediction"]["risk_score"], 0.72)

    def test_prefall_state(self):
        snap = {**self.snapshot, "state": "Medium Risk Detected", "riskPercent": 55}
        result = serialize_status(snap)
        self.assertEqual(result["prediction"]["state"], "Pre-fall")
        self.assertEqual(result["prediction"]["business_state"], "warning")

    def test_confirmed_prefall_never_serializes_below_warning_floor(self):
        snap = {**self.snapshot, "state": "Pre-fall", "riskPercent": 12}
        result = serialize_status(snap)
        self.assertAlmostEqual(result["prediction"]["risk_score"], 0.45)

    def test_recovery_maps_to_normal_instead_of_unknown(self):
        snap = {**self.snapshot, "state": "Recovery", "riskPercent": 10}
        result = serialize_status(snap)
        self.assertEqual(result["prediction"]["state"], "Normal")
        self.assertEqual(result["prediction"]["business_state"], "safe")

    def test_unknown_state_when_person_not_visible(self):
        snap = {**self.snapshot, "state": "Person Not Visible"}
        result = serialize_status(snap)
        self.assertEqual(result["prediction"]["state"], "Unknown")

    def test_not_monitoring(self):
        snap = {**self.snapshot, "running": False, "state": "Idle", "fps": 0.0}
        result = serialize_status(snap)
        self.assertFalse(result["monitoring"])
        self.assertEqual(result["performance"]["fps"], 0.0)
        self.assertEqual(result["prediction"]["risk_score"], 0.0)
        self.assertEqual(result["prediction"]["confidence"], 0.0)

    def test_risk_score_clamped(self):
        snap = {**self.snapshot, "riskPercent": 150, "confidencePercent": -10}
        result = serialize_status(snap)
        self.assertAlmostEqual(result["prediction"]["risk_score"], 1.0)
        self.assertAlmostEqual(result["prediction"]["confidence"], 0.0)

    def test_sequence_increments(self):
        from fall_prediction_service.serialization import reset_sequence
        reset_sequence(0)
        r1 = serialize_status(self.snapshot)
        r2 = serialize_status(self.snapshot)
        self.assertEqual(r1["sequence"], 1)
        self.assertEqual(r2["sequence"], 2)

    def test_has_timestamp(self):
        result = serialize_status(self.snapshot)
        self.assertGreater(result["timestamp_ms"], 0)

    def test_error_field_included_when_present(self):
        snap = {**self.snapshot, "error": "Camera not found"}
        result = serialize_status(snap)
        self.assertIsNotNone(result["error"])


class TestSerializeMonitorCommand(unittest.TestCase):
    def test_ok_start(self):
        result = serialize_monitor_command(ok=True, monitoring=True, session_id="abc123")
        self.assertTrue(result["ok"])
        self.assertTrue(result["monitoring"])
        self.assertEqual(result["session_id"], "abc123")

    def test_ok_stop(self):
        result = serialize_monitor_command(ok=True, monitoring=False)
        self.assertFalse(result["monitoring"])


class TestSerializeSettings(unittest.TestCase):
    class FakeSettings:
        sensitivity = "medium"
        camera_index = 0
        theme = "system"
        lang = "en"
        sound_alert = True

        def thresholds(self):
            return {"prefall": 0.45, "fall": 0.72}

    def test_basic(self):
        result = serialize_settings(self.FakeSettings())
        self.assertEqual(result["sensitivity"], "medium")
        self.assertEqual(result["camera_index"], 0)
        self.assertTrue(result["sound_alert"])
        self.assertIn("thresholds", result)


class TestSerializeProfile(unittest.TestCase):
    def test_from_to_dict(self):
        profile = type("Profile", (), {
            "to_dict": lambda self: {
                "id": "abc", "name": "Test", "createdAt": "2025-01-01",
                "fallCount": 3,
            }
        })()
        result = serialize_profile(profile)
        self.assertEqual(result["id"], "abc")
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["fallCount"], 3)

    def test_from_dict_row(self):
        result = serialize_profile({
            "id": "xyz", "name": "Row", "created_at": "2025-02-02",
        })
        self.assertEqual(result["id"], "xyz")
        self.assertEqual(result["name"], "Row")


class TestSerializeEvent(unittest.TestCase):
    def test_basic(self):
        row = {
            "id": "evt1",
            "event_type": "fall",
            "status": "open",
            "peak_risk": 0.92,
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": None,
            "session_id": "sess1",
        }
        result = serialize_event(row)
        self.assertEqual(result["id"], "evt1")
        self.assertEqual(result["event_type"], "fall")
        self.assertAlmostEqual(result["peak_risk"], 0.92)


class TestSerializePaginated(unittest.TestCase):
    def test_with_more(self):
        result = serialize_paginated(
            [{"id": "1"}, {"id": "2"}],
            next_cursor="cursor_abc",
            has_more=True,
        )
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["next_cursor"], "cursor_abc")
        self.assertTrue(result["has_more"])

    def test_last_page(self):
        result = serialize_paginated([{"id": "1"}], has_more=False)
        self.assertIsNone(result["next_cursor"])
        self.assertFalse(result["has_more"])


class TestServiceErrorCodes(unittest.TestCase):
    def test_unauthorized(self):
        err = unauthorized()
        self.assertEqual(err.code, "UNAUTHORIZED")
        self.assertFalse(err.retryable)

    def test_service_not_ready(self):
        err = service_not_ready()
        self.assertEqual(err.code, "SERVICE_NOT_READY")
        self.assertTrue(err.retryable)

    def test_monitor_already_running(self):
        err = monitor_already_running()
        self.assertEqual(err.code, "MONITOR_ALREADY_RUNNING")

    def test_all_error_codes_unique(self):
        codes = [
            unauthorized().code,
            service_not_ready().code,
            camera_in_use().code,
            monitor_already_running().code,
            import_conflict().code,
            invalid_argument().code,
            model_load_failed().code,
            database_error().code,
            not_found().code,
        ]
        self.assertEqual(len(codes), len(set(codes)))

    def test_error_to_dict(self):
        err = unauthorized("Bad token")
        d = err.to_dict()
        self.assertEqual(d["code"], "UNAUTHORIZED")
        self.assertEqual(d["details"], "Bad token")

    def test_error_http_status(self):
        from http import HTTPStatus
        self.assertEqual(unauthorized().http_status(), HTTPStatus.UNAUTHORIZED)
        self.assertEqual(service_not_ready().http_status(), HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(not_found().http_status(), HTTPStatus.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
