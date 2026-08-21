from __future__ import annotations

import copy
import unittest

import data_model_v3


class DataModelV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events = data_model_v3.load_events_v3()
        cls.by_id = {event["id"]: event for event in cls.events}
        cls.profiles = data_model_v3.load_event_profiles_v3()
        cls.venues = data_model_v3.load_venues_v3()

    def test_v3_covers_all_events_and_reuses_shared_venue(self) -> None:
        self.assertEqual(30, len(self.events))
        self.assertEqual(30, len(self.profiles))
        self.assertEqual(29, len(self.venues))
        self.assertEqual("V001", self.by_id["001"]["venue_id"])
        self.assertEqual("V001", self.by_id["030"]["venue_id"])

    def test_experience_profile_distinguishes_seated_and_walking(self) -> None:
        opening = self.by_id["001"]["experience_profile"]
        walk = self.by_id["009"]["experience_profile"]
        self.assertEqual("mostly_seated", opening["posture"])
        self.assertEqual("guaranteed", opening["seating"])
        self.assertEqual("low", opening["mobility_load"])
        self.assertEqual("standing_or_walking", walk["posture"])
        self.assertEqual("none", walk["seating"])
        self.assertEqual("high", walk["mobility_load"])
        self.assertIn("walk_explore", walk["engagement_modes"])

    def test_duration_is_explicitly_bounded_and_sourced(self) -> None:
        opening = self.by_id["001"]
        exhibition = self.by_id["007"]
        self.assertEqual(
            {"typical_minutes": 150, "basis": "scheduled_program"},
            opening["estimated_visit_duration"],
        )
        self.assertEqual(
            {"typical_minutes": 75, "basis": "poc_authored"},
            exhibition["estimated_visit_duration"],
        )
        self.assertEqual(
            "rule_derived",
            opening["provenance_v3"]["estimated_visit_duration"]["derivation"],
        )

    def test_last_admission_does_not_invent_cutoff(self) -> None:
        fixed_start = self.by_id["001"]["last_admission"]
        open_entry = self.by_id["007"]["last_admission"]
        self.assertEqual({"time": None, "status": "not_applicable"}, fixed_start)
        self.assertEqual({"time": None, "status": "unknown"}, open_entry)
        self.assertFalse(data_model_v3.hard_filter_eligible(self.by_id["007"], "last_admission"))

    def test_unverified_geo_stays_unknown_and_not_routable(self) -> None:
        for venue in self.venues.values():
            geo = venue["geo"]
            self.assertIsNone(geo["latitude"])
            self.assertIsNone(geo["longitude"])
            self.assertEqual("unknown", geo["precision"])
            self.assertFalse(geo["routing_eligible"])
            self.assertEqual("pending_verification", geo["enrichment_status"])

    def test_field_level_provenance_controls_hard_filtering(self) -> None:
        event = self.by_id["001"]
        self.assertTrue(data_model_v3.hard_filter_eligible(event, "event_status"))
        self.assertTrue(data_model_v3.hard_filter_eligible(event, "experience_profile"))
        self.assertTrue(data_model_v3.hard_filter_eligible(event, "estimated_visit_duration"))

    def test_llm_inference_can_never_be_hard_filter_eligible(self) -> None:
        raw = {
            "source_type": "other",
            "source_ref": "llm:test",
            "derivation": "llm_inferred",
            "hard_filter_eligible": True,
            "note": "test",
        }
        with self.assertRaises(data_model_v3.DataModelV3Error):
            data_model_v3.validate_fact_provenance(raw)

    def test_partial_coordinates_are_rejected(self) -> None:
        raw = copy.deepcopy(self.venues["V001"])
        raw["geo"]["latitude"] = 33.0
        with self.assertRaises(data_model_v3.DataModelV3Error):
            data_model_v3.validate_venue_v3(raw)

    def test_unverified_coordinates_cannot_be_routing_eligible(self) -> None:
        raw = copy.deepcopy(self.venues["V001"])
        raw["geo"].update(
            {
                "latitude": 33.0,
                "longitude": 132.0,
                "precision": "venue_exact",
                "routing_eligible": True,
                "enrichment_status": "pending_verification",
            }
        )
        with self.assertRaises(data_model_v3.DataModelV3Error):
            data_model_v3.validate_venue_v3(raw)

    def test_profile_coverage_is_exact(self) -> None:
        profiles = dict(self.profiles)
        profiles.pop("030")
        with self.assertRaises(data_model_v3.DataModelV3Error):
            data_model_v3.compose_events_v3(
                [
                    {key: value for key, value in event.items() if key not in {
                        "data_model_version", "event_status", "venue_id", "venue_v3",
                        "experience_profile", "estimated_visit_duration", "last_admission",
                        "provenance_v3",
                    }}
                    for event in self.events
                ],
                profiles,
                self.venues,
            )

    def test_schema_rejects_unknown_fields(self) -> None:
        raw = copy.deepcopy(self.profiles["001"])
        raw["guessed_comfort"] = "high"
        with self.assertRaises(data_model_v3.DataModelV3Error):
            data_model_v3.validate_event_profile_v3(raw)


if __name__ == "__main__":
    unittest.main()
