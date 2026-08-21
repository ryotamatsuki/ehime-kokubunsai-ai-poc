from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent_models
import agent_orchestrator
import agent_planner
import agent_tools
import command_models
import conversation_router
import data_model_v3
import event_details
import event_pair_recommendation
import event_search
import experience_matcher
import experience_preferences
from app_config import POC_REFERENCE_DATE


PARSER_CASES = (
    ("座って楽しめるイベントある？", "seated", "required"),
    ("座って見たい", "seated", "required"),
    ("着席して楽しめるもの", "seated", "required"),
    ("立ちっぱなしじゃないもの", "seated", "required"),
    ("立ちっぱなしは嫌", "seated", "required"),
    ("座って楽しみたい", "seated", "required"),
    ("座って見られるもの", "seated", "required"),
    ("あまり歩きたくない", "low_mobility", "required"),
    ("歩かなくていいもの", "low_mobility", "required"),
    ("移動が少ないもの", "low_mobility", "required"),
    ("歩く距離が少ないもの", "low_mobility", "required"),
    ("歩かないもの", "low_mobility", "required"),
    ("なるべく歩きたくない", "low_mobility", "preferred"),
    ("できたら歩かないもの", "low_mobility", "preferred"),
    ("足が疲れにくそうなもの", "low_mobility", "preferred"),
    ("見るだけで楽しみたい", "watch_listen", "required"),
    ("見て楽しめるもの", "watch_listen", "required"),
    ("聞いて楽しめるもの", "watch_listen", "required"),
    ("鑑賞中心がいい", "watch_listen", "required"),
    ("見るだけ", "watch_listen", "required"),
    ("聞くだけ", "watch_listen", "required"),
    ("何か作ってみたい", "hands_on", "required"),
    ("何か作りたい", "hands_on", "required"),
    ("作ってみたい", "hands_on", "required"),
    ("体験できるもの", "hands_on", "required"),
    ("体験したい", "hands_on", "required"),
    ("ワークショップある？", "hands_on", "required"),
    ("体験型がいい", "hands_on", "required"),
    ("手を動かして楽しみたい", "hands_on", "required"),
    ("まち歩きしたい", "walk_explore", "required"),
    ("まち歩き", "walk_explore", "required"),
    ("散策できるイベント", "walk_explore", "required"),
    ("散策したい", "walk_explore", "required"),
    ("歩いて巡るもの", "walk_explore", "required"),
    ("歩いて巡りたい", "walk_explore", "required"),
    ("参加型", "audience_participation", "required"),
    ("一緒に参加", "audience_participation", "required"),
    ("観客参加", "audience_participation", "required"),
    ("参加して楽しめる", "audience_participation", "required"),
    ("みんなで参加", "audience_participation", "required"),
    ("できれば座りたい", "seated", "preferred"),
    ("なるべく座れるもの", "seated", "preferred"),
    ("座れる方がいい", "seated", "preferred"),
    ("できたら歩かないもの", "low_mobility", "preferred"),
    ("できるだけ座って見たい", "seated", "preferred"),
    ("座ってなくてもいい", "seated", "released"),
    ("座れなくてもいい", "seated", "released"),
    ("座っていなくてもいい", "seated", "released"),
    ("歩くイベントでも大丈夫", "low_mobility", "released"),
    ("歩いてもいい", "low_mobility", "released"),
    ("体験型じゃなくていい", "hands_on", "released"),
    ("参加型じゃなくていい", "audience_participation", "released"),
    ("歩くイベントは嫌", "walk_explore", "excluded"),
    ("まち歩きは除いて", "walk_explore", "excluded"),
    ("まち歩き以外", "walk_explore", "excluded"),
    ("座るイベント以外", "seated", "excluded"),
    ("体験型以外", "hands_on", "excluded"),
    ("体験型じゃないもの", "hands_on", "excluded"),
    ("まち歩きじゃないもの", "walk_explore", "excluded"),
)


class ExperiencePreferencesTests(unittest.TestCase):
    def test_parser_matrix_has_at_least_50_user_cases(self) -> None:
        self.assertGreaterEqual(len(PARSER_CASES), 50)
        for query, concept_id, strength in PARSER_CASES:
            with self.subTest(query=query):
                parsed = event_search.parse_query(query)
                all_ids = (
                    parsed.experience_required
                    + parsed.experience_preferred
                    + parsed.experience_excluded
                )
                if strength == "released":
                    self.assertNotIn(concept_id, all_ids)
                    self.assertEqual(parsed.soft_terms, [])
                else:
                    values = getattr(parsed, f"experience_{strength}")
                    self.assertIn(concept_id, values)

    def test_experience_language_is_not_legacy_soft_terms(self) -> None:
        queries = [query for query, _, _ in PARSER_CASES]
        queries += [
            "松山で座って楽しめるもの",
            "無料で座って見られるもの",
            "屋内で何か作れるもの",
            "予約不要で体験型",
            "雨でもあまり歩かないもの",
        ]
        for query in queries:
            with self.subTest(query=query):
                parsed = event_search.parse_query(query)
                self.assertFalse(
                    any(experience_preferences.is_experience_phrase(term) for term in parsed.soft_terms)
                )

    def test_vocabulary_is_closed_and_command_slots_are_strict(self) -> None:
        expected = {
            "seated",
            "low_mobility",
            "watch_listen",
            "hands_on",
            "walk_explore",
            "audience_participation",
        }
        self.assertEqual(set(experience_preferences.valid_concept_ids()), expected)
        slots = command_models.CommandSlots.from_dict(
            {
                "experience_required": ["seated"],
                "experience_preferred": ["low_mobility"],
                "experience_excluded": ["walk_explore"],
            }
        )
        self.assertEqual(slots.experience_required, ("seated",))
        with self.assertRaises(command_models.CommandValidationError):
            command_models.CommandSlots.from_dict({"experience_required": ["posture=mostly_seated"]})
        with self.assertRaises(command_models.CommandValidationError):
            command_models.CommandSlots.from_dict({"experience_required": ["seated", "seated"]})
        with self.assertRaises(command_models.CommandValidationError):
            command_models.CommandSlots.from_dict(
                {"experience_required": ["seated"], "experience_excluded": ["seated"]}
            )
        with self.assertRaises(command_models.CommandValidationError):
            command_models.CommandSlots.from_dict({"topics": ["座って"]})

    def test_malformed_vocabulary_is_rejected_as_a_validation_error(self) -> None:
        malformed = {
            "schema_version": 1,
            "concepts": [
                {
                    "id": "seated",
                    "label": "座って楽しめる",
                    "aliases": [{"unexpected": "object"}],
                    "predicate": {"posture": ["mostly_seated"]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            path.write_text(json.dumps(malformed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(experience_preferences.ExperienceVocabularyError):
                experience_preferences.load_vocabulary(path)
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(experience_preferences.ExperienceVocabularyError):
                experience_preferences.load_vocabulary(path)

    def test_profile_predicates_use_v3_expected_sets(self) -> None:
        events = event_search.load_events()
        expected = {
            "seated": {"001", "003", "013", "023", "030"},
            "low_mobility": {"001", "003", "013", "014", "023", "030"},
            "walk_explore": {"002", "004", "009", "015", "016", "017", "019", "020", "022", "025"},
            "hands_on": {"004", "005", "010", "011", "012", "014", "018", "021", "022", "026", "027", "028", "029"},
            "audience_participation": {"002", "006", "012", "018", "024", "026"},
        }
        for concept_id, expected_ids in expected.items():
            actual = {
                str(event["id"])
                for event in events
                if experience_matcher.concept_match(event, concept_id) is True
            }
            with self.subTest(concept_id=concept_id):
                self.assertEqual(actual, expected_ids)

    def test_seating_available_does_not_mean_seated(self) -> None:
        event = next(event for event in event_search.load_events() if event["id"] == "007")
        self.assertEqual(experience_matcher.concept_match(event, "seated"), False)
        self.assertFalse(experience_matcher.matches_experience(event, required=["seated"]))

    def test_unknown_and_inferred_facts_are_not_hard_matches(self) -> None:
        base = next(event for event in data_model_v3.load_events_v3() if event["id"] == "001")

        unknown = deepcopy(base)
        unknown["experience_profile"]["posture"] = "unknown"
        self.assertIsNone(experience_matcher.concept_match(unknown, "seated"))
        self.assertFalse(experience_matcher.matches_experience(unknown, required=["seated"]))
        self.assertTrue(experience_matcher.matches_experience(unknown, excluded=["seated"]))
        self.assertEqual(experience_matcher.preferred_match_count(unknown, ["seated"]), 0)

        inferred = deepcopy(base)
        inferred["provenance_v3"]["experience_profile"]["derivation"] = "llm_inferred"
        inferred["provenance_v3"]["experience_profile"]["hard_filter_eligible"] = False
        self.assertIsNone(experience_matcher.concept_match(inferred, "seated"))
        self.assertFalse(experience_matcher.matches_experience(inferred, required=["seated"]))

        missing = {"id": "999", "イベント名": "未登録"}
        self.assertIsNone(experience_matcher.concept_match(missing, "seated"))
        self.assertFalse(experience_matcher.matches_experience(missing, required=["seated"]))

    def test_unknown_selected_experience_is_not_summarized_as_a_fact(self) -> None:
        base = next(event for event in data_model_v3.load_events_v3() if event["id"] == "001")
        unknown = deepcopy(base)
        unknown["experience_profile"]["posture"] = "unknown"
        self.assertIn(
            "確認できません",
            experience_matcher.describe_event_experience(unknown, "このイベント座って見られる？"),
        )

    def test_matcher_ignores_event_name_and_genre_for_experience(self) -> None:
        event = next(event for event in data_model_v3.load_events_v3() if event["id"] == "001")
        changed = deepcopy(event)
        changed["イベント名"] = "文化財ウォークという名前に変更"
        changed["ジャンル"] = "自然"
        self.assertTrue(experience_matcher.matches_experience(changed, required=["seated"]))

    def test_required_excluded_and_preferred_semantics(self) -> None:
        seated = event_search.search_events("座って楽しめるイベントある？")
        self.assertEqual(set(seated.all_event_ids), {"001", "003", "013", "023", "030"})
        self.assertEqual(seated.filters.soft_terms, [])

        preferred = event_search.search_events("できれば座って楽しみたい")
        self.assertEqual(preferred.filters.experience_preferred, ["seated"])
        self.assertEqual(set(preferred.all_event_ids[:5]), {"001", "003", "013", "023", "030"})
        self.assertEqual(preferred.total_matches, 30)

        excluded = event_search.search_events("まち歩きは除いて")
        walking = {
            "002", "004", "009", "015", "016", "017", "019", "020", "022", "025"
        }
        self.assertTrue(walking.isdisjoint(set(excluded.all_event_ids)))

    def test_compound_conditions_remain_and_filters(self) -> None:
        cases = (
            ("松山で座って楽しめるもの", "seated"),
            ("5歳とあまり歩かなくていいもの", "low_mobility"),
            ("無料で座って見られるもの", "seated"),
            ("松山で予約不要で座れるもの", "seated"),
            ("雨の日でもあまり歩かないもの", "low_mobility"),
            ("屋内で何か作れるもの", "hands_on"),
            ("予約不要で体験型", "hands_on"),
        )
        for query, concept_id in cases:
            result = event_search.search_events(query)
            with self.subTest(query=query):
                self.assertTrue(
                    all(experience_matcher.concept_match(event, concept_id) is True for event in result.events),
                    result.message,
                )
                self.assertEqual(result.filters.soft_terms, [])
        self.assertEqual(
            set(event_search.search_events("無料で座って見られるもの").all_event_ids),
            {"001", "030"},
        )

    def test_pair_candidates_apply_experience_hard_filters(self) -> None:
        events = event_search.load_events()
        seated = next(event for event in events if event["id"] == "001")
        walking = next(event for event in events if event["id"] == "009")
        filters = {"experience_required": ["seated"]}
        self.assertTrue(
            event_pair_recommendation._matches_filters(
                seated, filters, (), POC_REFERENCE_DATE
            )
        )
        self.assertFalse(
            event_pair_recommendation._matches_filters(
                walking, filters, (), POC_REFERENCE_DATE
            )
        )

    def test_hard_experience_is_never_relaxed(self) -> None:
        result = event_search.search_events("11/4に座って楽しめるもの")
        self.assertEqual(result.total_matches, 0)
        self.assertEqual(result.near_matches, [])
        self.assertIsNone(result.relaxed_condition)
        self.assertEqual(result.all_near_event_ids, [])
        self.assertNotIn("009", result.all_event_ids)
        self.assertIn("座って楽しめる", result.message or "")
        self.assertIn("立ったり歩いたり", result.message or "")

    def test_release_refinement_clears_previous_experience_constraints(self) -> None:
        initial = event_search.search_events("座って楽しめるイベントある？")
        released = event_search.search_events(
            "その中で座ってなくてもいい",
            previous_filters=initial.filters.to_dict(),
            inherit_previous=True,
        )
        self.assertEqual(released.filters.experience_required, [])
        self.assertEqual(released.filters.experience_preferred, [])
        self.assertEqual(released.filters.experience_excluded, [])
        self.assertEqual(released.intent, "needs_condition")

    def test_planner_and_agent_tools_preserve_experience_filters(self) -> None:
        fallback = agent_planner.fallback_search_plan("座って楽しめるイベントある？")
        self.assertEqual(fallback.searches[0].filters["experience_required"], ["seated"])
        self.assertNotIn("soft_terms", fallback.searches[0].filters)
        self.assertFalse(fallback.allow_replan)
        self.assertTrue(agent_planner.validate_search_plan(fallback.to_dict()))

        previous = agent_models.SearchPlan(
            intent="discover",
            answer_type="list",
            searches=(
                agent_models.SearchSpec(
                    search_id="s1",
                    tool="search_events",
                    purpose="exact",
                    filters={"experience_required": ["seated"], "soft_terms": ["工芸"]},
                ),
            ),
            allow_replan=True,
        )
        removed_experience = previous.to_dict()
        removed_experience["searches"][0]["search_id"] = "s1-relaxed"
        removed_experience["searches"][0]["purpose"] = "relaxed"
        removed_experience["searches"][0]["relaxed"] = True
        removed_experience["searches"][0]["relaxed_fields"] = ["soft_terms"]
        removed_experience["searches"][0]["filters"] = {}
        self.assertIsNone(agent_planner.validate_replan_plan(removed_experience, previous))

        retained = deepcopy(removed_experience)
        retained["searches"][0]["filters"] = {"experience_required": ["seated"]}
        self.assertIsNotNone(agent_planner.validate_replan_plan(retained, previous))

        spec = agent_models.SearchSpec(
            search_id="experience",
            tool="search_events",
            purpose="exact",
            filters={"experience_required": ["seated"]},
        )
        tool_result = agent_tools.execute_structured_search(spec)
        self.assertEqual(set(tool_result.all_event_ids), {"001", "003", "013", "023", "030"})

        release_plan = agent_planner.fallback_search_plan("座ってなくてもいい")
        self.assertEqual(release_plan.searches[0].filters, {"dates": ["1900-01-01"]})
        response = agent_orchestrator.handle_agentic_query("座ってなくてもいい", {})
        self.assertEqual(response.total_matches, 0)
        self.assertFalse(response.planner_used)

    def test_deterministic_parser_wins_over_conflicting_valid_llm_ids(self) -> None:
        model_plan = {
            "intent": "discover",
            "answer_type": "list",
            "searches": [
                {
                    "search_id": "s1",
                    "tool": "search_events",
                    "purpose": "exact",
                    "filters": {
                        "experience_required": ["walk_explore"],
                        "soft_terms": ["座って"],
                    },
                    "relaxed": False,
                    "relaxed_fields": [],
                }
            ],
            "confidence": "high",
            "allow_replan": True,
        }
        with patch("agent_planner._call_modal_json", return_value=model_plan):
            plan = agent_planner.request_search_plan(
                {"query": "座って楽しめるイベントある？"}
            )
        filters = plan.searches[0].filters
        self.assertEqual(filters["experience_required"], ["seated"])
        self.assertNotIn("soft_terms", filters)
        self.assertFalse(plan.allow_replan)

    def test_command_adapter_keeps_typed_slots(self) -> None:
        slots = command_models.CommandSlots(
            municipalities=("松山市",),
            experience_required=("seated",),
            experience_preferred=("low_mobility",),
        )
        filters = event_search.command_slots_to_search_filters(slots)
        self.assertEqual(filters.experience_required, ["seated"])
        self.assertEqual(filters.experience_preferred, ["low_mobility"])
        self.assertEqual(filters.soft_terms, [])

    def test_selected_event_questions_use_v3_facts(self) -> None:
        results = event_search.search_events("座って楽しめるイベントある？").events
        for query, index, expected_field in (
            ("1番目は座れる？", 0, "experience_profile"),
            ("2番目って歩く？", 1, "experience_profile"),
            ("そのイベントは体験型？", None, "experience_profile"),
        ):
            route = conversation_router.route_conversation(
                query,
                results,
                results[0],
                {},
                POC_REFERENCE_DATE,
            )
            with self.subTest(query=query):
                self.assertEqual(route.detail_field, expected_field)
                self.assertIn(route.action_type, {"reference_followup", "detail_followup"})
                selected = route.selected_event or results[0]
                answer = event_details.answer_event_detail(selected, route.detail_field or "", query)
                self.assertTrue(answer)
                self.assertNotIn("推測", answer)
                if index is not None:
                    self.assertEqual(selected["id"], results[index]["id"])

        direct = conversation_router.route_conversation(
            "このイベント座って見られる？",
            [],
            results[0],
            {},
            POC_REFERENCE_DATE,
        )
        self.assertEqual(direct.action_type, "detail_followup")
        self.assertEqual(direct.detail_field, "experience_profile")

    def test_experience_facts_are_exposed_in_agentic_trace(self) -> None:
        parsed = event_search.parse_query("座って楽しめるイベントある？")
        coverage = agent_orchestrator.assess_parser_coverage(
            "座って楽しめるイベントある？", parsed
        )
        self.assertTrue(coverage.complete)
        self.assertIn("experience_required=seated", coverage.recognized_constraints)
        self.assertTrue(agent_orchestrator.parser_confidence_is_high(parsed, "座って楽しめるイベントある？"))


if __name__ == "__main__":
    unittest.main()
