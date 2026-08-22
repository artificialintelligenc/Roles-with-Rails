import unittest
from unittest.mock import patch

import numpy as np
import torch

from sero.config import SeroConfig
from sero.openrouter_client import OpenRouterSampleSkipError
from sero.role_card import RoleCard
from sero.trainer import SeroTrainer


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        return np.zeros((len(texts), 2), dtype=np.float32)


class FakeController:
    def train(self):
        return None

    def __call__(self, **kwargs):
        op_logits = torch.zeros(3, dtype=torch.float32, requires_grad=True)
        target_scores = torch.zeros(2, dtype=torch.float32, requires_grad=True)
        return op_logits, target_scores


def _make_trainer(**config_overrides):
    config = SeroConfig(
        openrouter_api_key="test-key",
        batch_size=1,
        warmup_epochs=0,
        main_epochs=1,
        **config_overrides,
    )
    seed_pool = [
        RoleCard(name="Base Role", system_prompt="base prompt"),
    ]
    task = {"id": "task-1", "prompt": "dummy prompt", "eval_fn": lambda _: 0.0}
    trainer = SeroTrainer(
        config=config,
        seed_pool=seed_pool,
        tasks=[task],
        client=None,
        encoder=FakeEncoder(),
    )
    trainer.controller = FakeController()
    return trainer, task


def _make_trainer_with_roles(role_names, p_min=2, **config_overrides):
    config = SeroConfig(
        openrouter_api_key="test-key",
        batch_size=1,
        warmup_epochs=0,
        main_epochs=1,
        p_min=p_min,
        **config_overrides,
    )
    seed_pool = [
        RoleCard(name=role_name, system_prompt=f"{role_name} prompt")
        for role_name in role_names
    ]
    task = {"id": "task-1", "prompt": "dummy prompt", "eval_fn": lambda _: 0.0}
    trainer = SeroTrainer(
        config=config,
        seed_pool=seed_pool,
        tasks=[task],
        client=None,
        encoder=FakeEncoder(),
    )
    trainer.controller = FakeController()
    return trainer, task


def _make_trainer_with_seed_pool(seed_pool, **config_overrides):
    config = SeroConfig(
        openrouter_api_key="test-key",
        batch_size=1,
        warmup_epochs=0,
        main_epochs=1,
        **config_overrides,
    )
    task = {"id": "task-1", "prompt": "dummy prompt", "eval_fn": lambda _: 0.0}
    trainer = SeroTrainer(
        config=config,
        seed_pool=seed_pool,
        tasks=[task],
        client=None,
        encoder=FakeEncoder(),
    )
    trainer.controller = FakeController()
    return trainer, task


def _make_trainer_with_task_count(task_count, **config_overrides):
    config_kwargs = {
        "openrouter_api_key": "test-key",
        "batch_size": max(task_count + 1, 1),
        "warmup_epochs": 0,
        "main_epochs": 1,
    }
    config_kwargs.update(config_overrides)
    config = SeroConfig(**config_kwargs)
    seed_pool = [RoleCard(name="Base Role", system_prompt="base prompt")]
    tasks = [
        {"id": f"task-{idx}", "prompt": f"dummy prompt {idx}", "eval_fn": lambda _: 0.0}
        for idx in range(task_count)
    ]
    trainer = SeroTrainer(
        config=config,
        seed_pool=seed_pool,
        tasks=tasks,
        client=None,
        encoder=FakeEncoder(),
    )
    trainer.controller = FakeController()
    return trainer


def _make_fake_phase_a(before_score, after_score):
    class FakePhaseA:
        def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
            self.pool = list(pool)

        def run(self, task_prompt, loo_target=None):
            score = after_score if len(self.pool) > 1 else before_score
            answer = f"answer-with-{len(self.pool)}-roles"
            return {
                "answer": answer,
                "score": score,
                "pool_mean_emb": np.zeros(2, dtype=np.float32),
                "fast_credits": {},
                "topology": {},
            }

    return FakePhaseA


def _make_fake_phase_a_by_pool_size(score_by_pool_size):
    class FakePhaseA:
        def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
            self.pool = list(pool)

        def run(self, task_prompt, loo_target=None):
            pool_size = len(self.pool)
            score = score_by_pool_size.get(pool_size, 0.0)
            answer = f"answer-with-{pool_size}-roles"
            return {
                "answer": answer,
                "score": score,
                "pool_mean_emb": np.zeros(2, dtype=np.float32),
                "fast_credits": {},
                "topology": {},
            }

    return FakePhaseA


def _make_counting_phase_a(score=0.5):
    class FakePhaseA:
        run_calls = []

        def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
            self.pool = list(pool)

        def run(self, task_prompt, loo_target=None):
            FakePhaseA.run_calls.append(len(self.pool))
            return {
                "answer": f"answer-with-{len(self.pool)}-roles",
                "score": score,
                "pool_mean_emb": np.zeros(2, dtype=np.float32),
                "fast_credits": {},
                "topology": {"pool_size": len(self.pool)},
            }

    return FakePhaseA


class TrainerAcceptanceTest(unittest.TestCase):
    def test_gradient_step_returns_early_when_controller_reward_training_disabled(self):
        config = SeroConfig(
            openrouter_api_key="test-key",
            batch_size=1,
            warmup_epochs=0,
            main_epochs=1,
            controller_reward_training=False,
        )
        trainer = SeroTrainer(
            config=config,
            seed_pool=[RoleCard(name="Base Role", system_prompt="base prompt")],
            tasks=[],
            client=None,
            encoder=FakeEncoder(),
        )
        trainer._batch_log_probs = [torch.tensor(0.0, requires_grad=True)]
        trainer._batch_rewards = [1.0]
        trainer._batch_op_logits = [torch.tensor([0.0, 0.0, 0.0], requires_grad=True)]

        with patch.object(trainer.optimizer, "step") as mock_step:
            trainer._gradient_step()

        mock_step.assert_not_called()
        self.assertEqual(trainer._batch_log_probs, [])
        self.assertEqual(trainer._batch_rewards, [])
        self.assertEqual(trainer._batch_op_logits, [])

    def test_gradient_step_entropy_regularization_backprops_to_op_logits(self):
        config = SeroConfig(
            openrouter_api_key="test-key",
            batch_size=2,
            warmup_epochs=0,
            main_epochs=1,
            entropy_beta=0.5,
        )
        trainer = SeroTrainer(
            config=config,
            seed_pool=[RoleCard(name="Base Role", system_prompt="base prompt")],
            tasks=[],
            client=None,
            encoder=FakeEncoder(),
        )
        log_prob_a = torch.tensor(0.0, requires_grad=True)
        log_prob_b = torch.tensor(0.0, requires_grad=True)
        op_logits_a = torch.tensor([2.0, 0.0, -1.0], requires_grad=True)
        op_logits_b = torch.tensor([-1.0, 2.0, 0.0], requires_grad=True)
        trainer._batch_log_probs = [log_prob_a, log_prob_b]
        trainer._batch_rewards = [0.0, 0.0]
        trainer._batch_op_logits = [op_logits_a, op_logits_b]

        trainer._gradient_step()

        self.assertIsNotNone(op_logits_a.grad)
        self.assertIsNotNone(op_logits_b.grad)
        self.assertGreater(float(op_logits_a.grad.abs().sum()), 0.0)
        self.assertGreater(float(op_logits_b.grad.abs().sum()), 0.0)

    def test_train_preserves_task_order_when_shuffle_disabled(self):
        trainer = _make_trainer_with_task_count(4, shuffle_train_tasks=False, loo_refresh_interval=999)
        seen = []

        def fake_run_episode(task, controller_uses_credit_state, strict_improvement_only, allow_remove=True):
            seen.append(task["id"])

        trainer._run_episode = fake_run_episode
        trainer._save_checkpoint = lambda *args, **kwargs: None
        trainer._save_logs = lambda *args, **kwargs: None

        trainer.train()

        self.assertEqual(seen, ["task-0", "task-1", "task-2", "task-3"])

    def test_train_shuffles_task_order_by_default(self):
        trainer = _make_trainer_with_task_count(4, loo_refresh_interval=999)

        with patch("sero.trainer.np.random.shuffle") as mocked_shuffle:
            trainer._run_episode = lambda *args, **kwargs: None
            trainer._save_checkpoint = lambda *args, **kwargs: None
            trainer._save_logs = lambda *args, **kwargs: None
            trainer.train()

        mocked_shuffle.assert_called_once_with(trainer.tasks)

    def test_train_skips_content_inspection_blocked_sample(self):
        trainer = _make_trainer_with_task_count(1, shuffle_train_tasks=False, loo_refresh_interval=999)

        class FailingPhaseA:
            def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
                self.pool = list(pool)

            def run(self, task_prompt, loo_target=None):
                raise OpenRouterSampleSkipError("data_inspection_failed")

        trainer._save_checkpoint = lambda *args, **kwargs: None
        trainer._save_logs = lambda *args, **kwargs: None

        with patch("sero.trainer.PhaseA", FailingPhaseA):
            stats = trainer.train()

        self.assertEqual(stats.episodes, 1)
        self.assertEqual(stats.skip_count, 1)
        self.assertEqual(len(stats.episode_records), 1)
        rec = stats.episode_records[0]
        self.assertEqual(rec.task_id, "task-0")
        self.assertEqual(rec.executed_op_type, "skip_error")
        self.assertEqual(rec.reward, 0.0)
        self.assertEqual(rec.pool_before, ["Base Role"])
        self.assertEqual(rec.committed_pool_after, ["Base Role"])

    def test_naturalplan_role_gate_allows_trip_travel_time_language(self):
        trainer, _ = _make_trainer()
        task = {
            "id": "trip::demo",
            "benchmark": "naturalplan",
            "prompt": "Trip planning task. IMPORTANT: This is a TRIP PLANNING task.",
            "eval_fn": lambda _: 0.0,
        }
        card = RoleCard(
            name="Trip Flight Compatibility Checker",
            system_prompt=(
                "Analyze trip constraints, direct flights, city stays, shared flight-day arithmetic, "
                "and required travel times between cities. Identify incompatible flight schedules."
            ),
            capability_tags=["trip", "flight-analysis"],
            capability_family="trip-flight-checking",
            communication_protocol="output trip flight compatibility findings",
        )

        self.assertIsNone(trainer._domain_role_rejection(card, task))

    def test_naturalplan_role_gate_rejects_mixed_trip_meeting_specialist(self):
        trainer, _ = _make_trainer()
        task = {
            "id": "trip::demo",
            "benchmark": "naturalplan",
            "prompt": "Trip planning task. IMPORTANT: This is a TRIP PLANNING task.",
            "eval_fn": lambda _: 0.0,
        }
        card = RoleCard(
            name="Trip Meeting Optimizer",
            system_prompt=(
                "Analyze city stays and direct flights, then maximize feasible people meetings using "
                "availability windows and wait steps."
            ),
            capability_tags=["trip", "meeting"],
            capability_family="trip-meeting-mixed",
            communication_protocol="output a mixed trip and meeting plan",
        )

        rejection = trainer._domain_role_rejection(card, task)

        self.assertIsNotNone(rejection)
        self.assertIn("mixes domain-family semantics", rejection)

    def test_naturalplan_executor_context_keeps_task_prompt_first(self):
        trainer, _ = _make_trainer()
        task = {
            "id": "calendar::demo",
            "benchmark": "naturalplan",
            "prompt": (
                "Calendar scheduling task. IMPORTANT: This is a CALENDAR SCHEDULING task. "
                "Alex is busy Monday 09:00-10:00 and wants a 30-minute meeting."
            ),
            "eval_fn": lambda _: 0.0,
        }

        context = trainer._build_executor_task_context(task)

        self.assertTrue(context.startswith("Task prompt:\nCalendar scheduling task."))
        self.assertLess(
            context.index("Calendar scheduling task."),
            context.index("NaturalPlan SERO guardrails:"),
        )
        self.assertIn("CALENDAR SCHEDULING", context[:400])

    def test_main_zero_reward_edit_is_rejected_and_credit_state_reverted(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Tentative Role", system_prompt="tentative prompt")

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name, parent_ema=0.5)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.5)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role"])
        self.assertNotIn("Tentative Role", trainer.credit_engine.all_role_info())
        rec = trainer.stats.episode_records[-1]
        self.assertFalse(rec.accepted)
        self.assertEqual(rec.reward, 0.0)
        self.assertEqual(rec.executed_op_type, "add_anchor")
        self.assertEqual(rec.pool_before, ["Base Role"])
        self.assertEqual(rec.pool_after, ["Base Role", "Tentative Role"])
        self.assertEqual(rec.candidate_pool_after, ["Base Role", "Tentative Role"])
        self.assertEqual(rec.committed_pool_after, ["Base Role"])
        self.assertEqual(rec.pool_size, 2)
        self.assertEqual(rec.committed_pool_size, 1)
        self.assertIn("Tentative Role", rec.credit_snapshot)
        self.assertNotIn("Tentative Role", rec.committed_credit_snapshot)

    def test_rejected_edit_restores_fast_credit_state(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Tentative Role", system_prompt="tentative prompt")

        class FastCreditPhaseA:
            call_idx = 0

            def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
                self.pool = list(pool)
                self.credit_engine = credit_engine

            def run(self, task_prompt, loo_target=None, update_fast_credits=True):
                FastCreditPhaseA.call_idx += 1
                if FastCreditPhaseA.call_idx == 1:
                    if update_fast_credits:
                        self.credit_engine.update_fast_credit("Base Role", 0.11)
                    score = 0.5
                    fast_credit = 0.11
                else:
                    if update_fast_credits:
                        self.credit_engine.update_fast_credit("Base Role", 0.77)
                        self.credit_engine.update_fast_credit("Tentative Role", 0.66)
                    score = 0.4
                    fast_credit = 0.77
                return {
                    "answer": "answer",
                    "score": score,
                    "pool_mean_emb": np.zeros(2, dtype=np.float32),
                    "fast_credits": {"Base Role": fast_credit},
                    "topology": {},
                }

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", FastCreditPhaseA), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        rec = trainer.stats.episode_records[-1]
        self.assertFalse(rec.accepted)
        self.assertEqual(rec.fast_credits_before["Base Role"], 0.11)
        self.assertEqual(rec.fast_credits_after["Base Role"], 0.77)
        self.assertEqual(rec.credit_snapshot["Base Role"]["fast_credit"], 0.77)
        self.assertEqual(rec.committed_credit_snapshot["Base Role"]["fast_credit"], 0.11)
        self.assertEqual(trainer.credit_engine.get_fast_credit("Base Role"), 0.11)
        self.assertNotIn("Tentative Role", trainer.credit_engine.all_role_info())

    def test_rejected_real_edit_counts_as_committed_pool_unchanged(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Tentative Role", system_prompt="tentative prompt")

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name, parent_ema=0.5)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.5)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual(trainer.stats.noop_count, 0)
        self.assertEqual(trainer.stats.add_count, 1)
        self.assertEqual(
            trainer.stats.committed_pool_unchanged_rate(window=1, require_full_window=True),
            1.0,
        )

    def test_main_positive_reward_edit_is_accepted(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Helpful Role", system_prompt="helpful prompt")

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name, parent_ema=0.5)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.8)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role", "Helpful Role"])
        self.assertIn("Helpful Role", trainer.credit_engine.all_role_info())
        rec = trainer.stats.episode_records[-1]
        self.assertTrue(rec.accepted)
        self.assertAlmostEqual(rec.reward, 0.3, places=6)
        self.assertEqual(rec.executed_op_type, "add_anchor")
        self.assertEqual(rec.pool_after, ["Base Role", "Helpful Role"])
        self.assertEqual(rec.committed_pool_after, ["Base Role", "Helpful Role"])
        self.assertEqual(rec.pool_size, 2)
        self.assertEqual(rec.committed_pool_size, 2)

    def test_warmup_zero_reward_edit_is_still_accepted(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Warmup Role", system_prompt="warmup prompt")

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name, parent_ema=0.5)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.5)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=False, strict_improvement_only=False)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role", "Warmup Role"])
        self.assertIn("Warmup Role", trainer.credit_engine.all_role_info())
        rec = trainer.stats.episode_records[-1]
        self.assertTrue(rec.accepted)
        self.assertEqual(rec.reward, 0.0)
        self.assertEqual(rec.executed_op_type, "add_anchor")

    def test_warmup_negative_reward_edit_is_rejected(self):
        trainer, task = _make_trainer()
        added_role = RoleCard(name="Harmful Warmup Role", system_prompt="harmful prompt")

        def fake_apply_action(op_type, target, task_context):
            trainer.credit_engine.register_role(added_role.name, parent_ema=0.5)
            return trainer.pool + [added_role], None, "add_anchor"

        trainer._apply_action = fake_apply_action

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.4)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=False, strict_improvement_only=False)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role"])
        self.assertNotIn("Harmful Warmup Role", trainer.credit_engine.all_role_info())
        rec = trainer.stats.episode_records[-1]
        self.assertFalse(rec.accepted)
        self.assertLess(rec.reward, 0.0)
        self.assertEqual(rec.executed_op_type, "add_anchor")
        self.assertEqual(rec.pool_after, ["Base Role", "Harmful Warmup Role"])
        self.assertEqual(rec.committed_pool_after, ["Base Role"])

    def test_warmup_masks_remove_before_sampling(self):
        trainer, task = _make_trainer()
        observed = {}

        def fake_sample_action(op_logits, target_scores_per_op, pool, use_credit_state, config=None, required_capability_families=None):
            observed["remove_logit"] = float(op_logits[1].item())
            return "noop", None, torch.tensor(0.0, requires_grad=True)

        with patch("sero.trainer.PhaseA", _make_fake_phase_a(0.5, 0.5)), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch("sero.trainer.sample_action", side_effect=fake_sample_action):
            trainer._run_episode(task, controller_uses_credit_state=False, strict_improvement_only=False, allow_remove=False)

        self.assertTrue(np.isneginf(observed["remove_logit"]))

    def test_collapse_detector_requires_full_window(self):
        trainer = _make_trainer_with_task_count(23)

        def fake_run_episode(task, controller_uses_credit_state, strict_improvement_only, allow_remove=True):
            trainer.stats.record_op("noop", committed_pool_unchanged=True)

        saved = []

        def fake_save_checkpoint(epoch, tag="checkpoint"):
            saved.append((epoch, tag))

        with patch.object(trainer, "_run_episode", side_effect=fake_run_episode), patch.object(
            trainer, "_save_checkpoint", side_effect=fake_save_checkpoint
        ), patch.object(trainer, "_save_logs", return_value=None):
            stats = trainer.train()

        self.assertEqual(stats.episodes, 23)
        self.assertEqual(saved, [(0, "checkpoint")])
        self.assertIsNone(
            trainer.stats.committed_pool_unchanged_rate(window=24, require_full_window=True)
        )

    def test_collapse_detector_is_disabled_during_warmup(self):
        trainer = _make_trainer_with_task_count(24, warmup_epochs=1, main_epochs=0)

        def fake_run_episode(task, controller_uses_credit_state, strict_improvement_only, allow_remove=True):
            trainer.stats.record_op("noop", committed_pool_unchanged=True)

        saved = []

        def fake_save_checkpoint(epoch, tag="checkpoint"):
            saved.append((epoch, tag))

        with patch.object(trainer, "_run_episode", side_effect=fake_run_episode), patch.object(
            trainer, "_save_checkpoint", side_effect=fake_save_checkpoint
        ), patch.object(trainer, "_save_logs", return_value=None):
            stats = trainer.train()

        self.assertEqual(stats.episodes, 24)
        self.assertEqual(saved, [(0, "checkpoint")])
        self.assertEqual(
            trainer.stats.committed_pool_unchanged_rate(window=24, require_full_window=True),
            1.0,
        )

    def test_collapse_detector_uses_committed_pool_unchanged_rate(self):
        # collapse window = 24, and min episodes before collapse = max(tasks, 24).
        # Use exactly 24 tasks so the window can be reached within one epoch.
        trainer = _make_trainer_with_task_count(24)

        def fake_run_episode(task, controller_uses_credit_state, strict_improvement_only, allow_remove=True):
            trainer.stats.record_op("add_anchor", committed_pool_unchanged=True)

        saved = []

        def fake_save_checkpoint(epoch, tag="checkpoint"):
            saved.append((epoch, tag))

        with patch.object(trainer, "_run_episode", side_effect=fake_run_episode), patch.object(
            trainer, "_save_checkpoint", side_effect=fake_save_checkpoint
        ), patch.object(trainer, "_save_logs", return_value=None):
            stats = trainer.train()

        self.assertEqual(stats.episodes, 24)
        self.assertEqual(saved, [(0, "collapsed")])
        self.assertEqual(trainer.stats.noop_count, 0)
        self.assertEqual(trainer.stats.add_count, 24)
        self.assertEqual(
            trainer.stats.committed_pool_unchanged_rate(window=24, require_full_window=True),
            1.0,
        )

    def test_committed_remove_is_added_to_recent_history(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Remove Me", "Keep Me"])

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({3: 0.5, 2: 0.8})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("remove", "Remove Me", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role", "Keep Me"])
        self.assertEqual(trainer.recently_removed_role_names, ["Remove Me"])
        rec = trainer.stats.episode_records[-1]
        self.assertTrue(rec.accepted)
        self.assertEqual(rec.executed_op_type, "remove")

    def test_rejected_remove_does_not_enter_recent_history(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Remove Me", "Keep Me"])

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({3: 0.5, 2: 0.5})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("remove", "Remove Me", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual([role.name for role in trainer.pool], ["Base Role", "Remove Me", "Keep Me"])
        self.assertEqual(trainer.recently_removed_role_names, [])
        rec = trainer.stats.episode_records[-1]
        self.assertFalse(rec.accepted)
        self.assertEqual(rec.executed_op_type, "remove")
        self.assertEqual(rec.pool_after, ["Base Role", "Keep Me"])
        self.assertEqual(rec.committed_pool_after, ["Base Role", "Remove Me", "Keep Me"])
        self.assertEqual(rec.pool_size, 2)
        self.assertEqual(rec.committed_pool_size, 3)

    def test_add_episode_receives_recent_history_and_logs_prompt(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Keep Me"])
        trainer.recently_removed_role_names = ["Remove Me"]
        added_role = RoleCard(name="Helpful Role", system_prompt="helpful prompt")
        captured = {}

        def fake_add_anchor(*args, **kwargs):
            captured["recently_removed_role_names"] = list(kwargs["recently_removed_role_names"])
            return {
                "card": added_role,
                "anchor_name": "Base Role",
                "executor_prompt": "5. DO NOT recreate recently removed roles by exact name: [\"Remove Me\"]",
                "llm_raw_output": "{}",
                "fallback": False,
                "recently_removed_roles": list(kwargs["recently_removed_role_names"]),
            }

        with patch.object(trainer.executor, "add_anchor", side_effect=fake_add_anchor), patch(
            "sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.5, 3: 0.8})
        ), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("add_anchor", "Base Role", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual(captured["recently_removed_role_names"], ["Remove Me"])
        rec = trainer.stats.episode_records[-1]
        self.assertIn("Remove Me", rec.executor_prompt)
        self.assertEqual(rec.recent_removed_roles, ["Remove Me"])
        self.assertEqual(rec.executed_op_type, "add_anchor")

    def test_apply_action_passes_task_conditioning_to_executor(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Route Analyst", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Constraint Checker", system_prompt="validator prompt", capability_family="validation"),
            ],
            p_min=1,
        )
        trainer.pool = [trainer.pool[0]]
        trainer.recent_failed_task_examples = ["score=0.400 | task with route constraints"]
        trainer.recent_failure_patterns = ["route-coverage"]
        added_role = RoleCard(name="Fallback Analyst", system_prompt="fallback prompt", capability_family="validation")
        captured = {}

        def fake_add_anchor(*args, **kwargs):
            captured["task_conditioning"] = kwargs["task_conditioning"]
            return {
                "card": added_role,
                "anchor_name": "Route Analyst",
                "executor_prompt": "prompt",
                "llm_raw_output": "{}",
                "fallback": False,
                "recently_removed_roles": list(kwargs["recently_removed_role_names"]),
                "task_conditioning": kwargs["task_conditioning"],
            }

        with patch.object(trainer.executor, "add_anchor", side_effect=fake_add_anchor):
            pool_after, add_trace, executed_op_type = trainer._apply_action("add_anchor", "Route Analyst", "current task")

        self.assertEqual(executed_op_type, "add_anchor")
        self.assertEqual([role.name for role in pool_after], ["Route Analyst", "Fallback Analyst"])
        self.assertEqual(captured["task_conditioning"]["missing_capability_families"], ["validation"])
        self.assertEqual(captured["task_conditioning"]["recent_failed_task_examples"], ["score=0.400 | task with route constraints"])
        self.assertEqual(captured["task_conditioning"]["recent_failure_patterns"], ["route-coverage"])
        self.assertEqual(captured["task_conditioning"]["diversity_status"], "low")
        self.assertEqual(captured["task_conditioning"]["orthogonal_family_blacklist"], ["route"])
        self.assertEqual(captured["task_conditioning"]["allowed_new_role_types"], ["specialist"])
        self.assertIn("Do NOT create a new validator role", captured["task_conditioning"]["validator_role_policy"])
        self.assertEqual(add_trace["task_conditioning"]["missing_capability_families"], ["validation"])

    def test_apply_action_rejects_new_validator_when_conditional_validator_pass_enabled(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Route Analyst", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Constraint Checker", system_prompt="validator prompt", capability_family="validation", role_type="validator", protected=True),
            ],
            p_min=1,
            conditional_validator_pass=True,
        )
        validator_card = RoleCard(
            name="Extra Validator",
            system_prompt="validator prompt",
            capability_family="extra-validation",
            role_type="validator",
        )

        with patch.object(trainer.executor, "add_anchor", return_value={
            "card": validator_card,
            "anchor_name": "Route Analyst",
            "executor_prompt": "prompt",
            "llm_raw_output": "{}",
            "fallback": False,
            "recently_removed_roles": [],
            "task_conditioning": trainer._build_add_task_conditioning(trainer.pool),
        }):
            pool_after, add_trace, executed_op_type = trainer._apply_action("add_anchor", "Route Analyst", "task")

        self.assertEqual(executed_op_type, "invalid_add")
        self.assertEqual([role.name for role in pool_after], ["Route Analyst", "Constraint Checker"])
        self.assertIsNotNone(add_trace)
        self.assertEqual(add_trace["new_role_card"]["role_type"], "validator")
        self.assertEqual(add_trace["rejection_reason"], "validator-role-disabled")

    def test_apply_action_keeps_pool_frozen_when_freeze_role_pool_enabled(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [RoleCard(name="Route Analyst", system_prompt="route prompt", capability_family="route")],
            p_min=1,
            freeze_role_pool=True,
        )

        with patch.object(trainer.executor, "add_anchor", side_effect=AssertionError("designer should not run")):
            pool_after, add_trace, executed_op_type = trainer._apply_action("add_anchor", "Route Analyst", "task")

        self.assertEqual(executed_op_type, "noop")
        self.assertEqual([role.name for role in pool_after], ["Route Analyst"])
        self.assertIsNone(add_trace)

    def test_apply_action_rejects_low_diversity_add_that_reuses_dominant_family(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Route Analyst A", system_prompt="route prompt A", capability_family="route"),
                RoleCard(name="Route Analyst B", system_prompt="route prompt B", capability_family="route"),
            ],
            p_min=1,
        )
        duplicate_family_card = RoleCard(
            name="Route Analyst C",
            system_prompt="route prompt C",
            capability_family="route",
            role_type="specialist",
        )

        with patch.object(trainer.executor, "add_anchor", return_value={
            "card": duplicate_family_card,
            "anchor_name": "Route Analyst A",
            "executor_prompt": "prompt",
            "llm_raw_output": "{}",
            "fallback": False,
            "recently_removed_roles": [],
            "task_conditioning": trainer._build_add_task_conditioning(trainer.pool),
        }):
            pool_after, add_trace, executed_op_type = trainer._apply_action("add_anchor", "Route Analyst A", "task")

        self.assertEqual(executed_op_type, "invalid_add")
        self.assertEqual([role.name for role in pool_after], ["Route Analyst A", "Route Analyst B"])
        self.assertTrue(add_trace["rejection_reason"].startswith("orthogonal:"))

    def test_episode_record_captures_diversity_monitor_snapshot(self):
        trainer, task = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Route Analyst A", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Route Analyst B", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Constraint Checker", system_prompt="validator prompt", capability_family="validation"),
            ],
            p_min=1,
        )
        trainer.pool = trainer.pool[:2]

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.4})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("noop", None, torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        rec = trainer.stats.episode_records[-1]
        self.assertEqual(rec.missing_capability_families, ["validation"])
        self.assertEqual(rec.orthogonal_family_blacklist, ["route"])
        self.assertTrue(rec.low_diversity_mode)
        self.assertEqual(rec.family_diversity, 0.5)
        self.assertEqual(rec.family_dominance, 1.0)
        self.assertEqual(rec.prompt_similarity_mean, 0.0)
        self.assertEqual(rec.orthogonal_role_blacklist, ["Route Analyst A", "Route Analyst B"])

    def test_recent_removed_history_keeps_last_ten_roles(self):
        trainer, _ = _make_trainer_with_roles(["Base Role", "Keep Me"], p_min=1)

        for idx in range(12):
            trainer._remember_removed_role(f"Role {idx}")

        self.assertEqual(
            trainer.recently_removed_role_names,
            [f"Role {idx}" for idx in range(2, 12)],
        )

        trainer._remember_removed_role("Role 5")

        self.assertEqual(len(trainer.recently_removed_role_names), 10)
        self.assertEqual(trainer.recently_removed_role_names[-1], "Role 5")
        self.assertEqual(trainer.recently_removed_role_names.count("Role 5"), 1)

    def test_noop_reuses_before_result_and_reward_is_zero(self):
        trainer, task = _make_trainer()
        counting_phase_a = _make_counting_phase_a(score=0.5)

        with patch("sero.trainer.PhaseA", counting_phase_a), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("noop", None, torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual(counting_phase_a.run_calls, [1])
        rec = trainer.stats.episode_records[-1]
        self.assertEqual(rec.reward, 0.0)
        self.assertEqual(rec.executed_op_type, "noop")
        self.assertEqual(rec.score_before, rec.score_after)
        self.assertEqual(rec.topology_before, rec.topology_after)
        self.assertEqual(rec.topology_after, rec.committed_topology_after)
        self.assertEqual(rec.pool_after, rec.committed_pool_after)

    def test_invalid_remove_reuses_before_result_and_reward_is_zero(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Keep Me"], p_min=2)
        counting_phase_a = _make_counting_phase_a(score=0.5)

        with patch("sero.trainer.PhaseA", counting_phase_a), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch(
            "sero.trainer.sample_action",
            return_value=("remove", "Keep Me", torch.tensor(0.0, requires_grad=True)),
        ):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertEqual(counting_phase_a.run_calls, [2])
        rec = trainer.stats.episode_records[-1]
        self.assertEqual(rec.reward, 0.0)
        self.assertFalse(rec.accepted)
        self.assertEqual(rec.op_type, "remove")
        self.assertEqual(rec.executed_op_type, "invalid_remove")
        self.assertEqual(rec.pool_before, rec.pool_after)
        self.assertEqual(rec.pool_after, rec.committed_pool_after)
        self.assertEqual(rec.score_before, rec.score_after)

    def test_added_role_uses_zero_ema_and_configured_initial_updates(self):
        trainer, _ = _make_trainer_with_roles(
            ["Base Role", "Keep Me"],
            new_role_initial_n_updates=5,
        )
        added_role = RoleCard(name="Cold Start Role", system_prompt="cold start prompt")

        with patch.object(trainer.executor, "add_anchor", return_value={
            "card": added_role,
            "anchor_name": "Base Role",
            "executor_prompt": "prompt",
            "llm_raw_output": "{}",
            "fallback": False,
            "recently_removed_roles": [],
        }):
            pool_after, _, executed_op_type = trainer._apply_action("add_anchor", "Base Role", "task")

        self.assertEqual([role.name for role in pool_after], ["Base Role", "Keep Me", "Cold Start Role"])
        self.assertEqual(executed_op_type, "add_anchor")
        role_info = trainer.credit_engine.all_role_info(["Cold Start Role"])["Cold Start Role"]
        self.assertEqual(role_info["ema"], 0.0)
        self.assertEqual(role_info["recent_phi"], 0.0)
        self.assertEqual(role_info["n_updates"], 5)

    def test_pool_at_max_masks_add_before_sampling(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Keep Me"], p_min=1, p_max=2)
        observed = {}

        def fake_sample_action(op_logits, target_scores_per_op, pool, use_credit_state, config=None, required_capability_families=None):
            observed["add_logit"] = float(op_logits[0].item())
            observed["remove_logit"] = float(op_logits[1].item())
            return "noop", None, torch.tensor(0.0, requires_grad=True)

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.5})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch("sero.trainer.sample_action", side_effect=fake_sample_action):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertTrue(np.isneginf(observed["add_logit"]))
        self.assertFalse(np.isneginf(observed["remove_logit"]))

    def test_pool_at_min_masks_remove_before_sampling(self):
        trainer, task = _make_trainer_with_roles(["Base Role", "Keep Me"], p_min=2)
        observed = {}

        def fake_sample_action(op_logits, target_scores_per_op, pool, use_credit_state, config=None, required_capability_families=None):
            observed["add_logit"] = float(op_logits[0].item())
            observed["remove_logit"] = float(op_logits[1].item())
            return "noop", None, torch.tensor(0.0, requires_grad=True)

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.5})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch("sero.trainer.sample_action", side_effect=fake_sample_action):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertFalse(np.isneginf(observed["add_logit"]))
        self.assertTrue(np.isneginf(observed["remove_logit"]))

    def test_last_required_family_masks_remove_before_sampling(self):
        trainer, task = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Route Analyst", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Constraint Checker", system_prompt="validator prompt", capability_family="validation"),
            ],
            p_min=1,
        )
        observed = {}

        def fake_sample_action(op_logits, target_scores_per_op, pool, use_credit_state, config=None, required_capability_families=None):
            observed["remove_logit"] = float(op_logits[1].item())
            observed["required_families"] = set(required_capability_families or [])
            return "noop", None, torch.tensor(0.0, requires_grad=True)

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.5})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch("sero.trainer.sample_action", side_effect=fake_sample_action):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertTrue(np.isneginf(observed["remove_logit"]))
        self.assertEqual(observed["required_families"], {"route", "validation"})

    def test_last_required_validator_masks_remove_before_sampling(self):
        trainer, task = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Planner", system_prompt="planner prompt", capability_family="planning"),
                RoleCard(
                    name="Constraint Checker",
                    system_prompt="validator prompt",
                    capability_family="validation",
                    role_type="validator",
                ),
            ],
            p_min=1,
            validator_slots=1,
        )
        observed = {}

        def fake_sample_action(op_logits, target_scores_per_op, pool, use_credit_state, config=None, required_capability_families=None):
            observed["remove_logit"] = float(op_logits[1].item())
            return "noop", None, torch.tensor(0.0, requires_grad=True)

        with patch("sero.trainer.PhaseA", _make_fake_phase_a_by_pool_size({2: 0.5})), patch(
            "sero.trainer.build_controller_tensors",
            return_value={"credit_stats": torch.zeros(5, dtype=torch.float32)},
        ), patch("sero.trainer.sample_action", side_effect=fake_sample_action):
            trainer._run_episode(task, controller_uses_credit_state=True, strict_improvement_only=True)

        self.assertTrue(np.isneginf(observed["remove_logit"]))

    def test_apply_action_rejects_remove_of_last_required_validator(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Planner", system_prompt="planner prompt", capability_family="planning"),
                RoleCard(
                    name="Constraint Checker",
                    system_prompt="validator prompt",
                    capability_family="validation",
                    role_type="validator",
                ),
            ],
            p_min=1,
            validator_slots=1,
        )

        pool_after, add_trace, executed_op_type = trainer._apply_action("remove", "Constraint Checker", "task")

        self.assertEqual([role.name for role in pool_after], ["Planner", "Constraint Checker"])
        self.assertIsNone(add_trace)
        self.assertEqual(executed_op_type, "invalid_remove")

    def test_apply_action_rejects_remove_that_breaks_required_family_coverage(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Core Planner", system_prompt="planner prompt", capability_family="planning"),
                RoleCard(name="Route Analyst A", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Route Analyst B", system_prompt="route prompt", capability_family="route"),
            ],
            p_min=1,
        )

        pool_after, add_trace, executed_op_type = trainer._apply_action("remove", "Core Planner", "task")

        self.assertEqual([role.name for role in pool_after], ["Core Planner", "Route Analyst A", "Route Analyst B"])
        self.assertIsNone(add_trace)
        self.assertEqual(executed_op_type, "invalid_remove")

    def test_apply_action_allows_remove_when_family_still_covered(self):
        trainer, _ = _make_trainer_with_seed_pool(
            [
                RoleCard(name="Core Planner", system_prompt="planner prompt", capability_family="planning"),
                RoleCard(name="Route Analyst A", system_prompt="route prompt", capability_family="route"),
                RoleCard(name="Route Analyst B", system_prompt="route prompt", capability_family="route"),
            ],
            p_min=1,
        )

        pool_after, add_trace, executed_op_type = trainer._apply_action("remove", "Route Analyst B", "task")

        self.assertEqual([role.name for role in pool_after], ["Core Planner", "Route Analyst A"])
        self.assertIsNone(add_trace)
        self.assertEqual(executed_op_type, "remove")

    def test_apply_action_labels_invalid_add_when_pool_is_full(self):
        trainer, _ = _make_trainer_with_roles(["Base Role", "Keep Me"], p_min=1, p_max=2)

        pool_after, add_trace, executed_op_type = trainer._apply_action("add_anchor", "Base Role", "task")

        self.assertEqual([role.name for role in pool_after], ["Base Role", "Keep Me"])
        self.assertIsNone(add_trace)
        self.assertEqual(executed_op_type, "invalid_add")

    def test_loo_refresh_delegates_to_full_pool_helper(self):
        trainer, task = _make_trainer_with_roles(
            ["Base Role", "Keep Me", "Third Role", "Fourth Role"],
            p_min=2,
            loo_min_pool_size=2,
        )
        captured = {}

        class DummyPhaseA:
            def __init__(self, config, client, encoder, credit_engine, pool, eval_fn):
                self.config = config
                self.client = client
                self.encoder = encoder
                self.credit_engine = credit_engine
                self.pool = list(pool)
                self.eval_fn = eval_fn

        def fake_full_pool_loo_refresh(phase_a, task_items):
            captured["phase_a"] = phase_a
            captured["task_items"] = list(task_items)

        with patch("sero.trainer.PhaseA", DummyPhaseA), patch(
            "sero.trainer.full_pool_loo_refresh",
            side_effect=fake_full_pool_loo_refresh,
        ):
            trainer._loo_refresh(sample_tasks=1)

        self.assertIsInstance(captured["phase_a"], DummyPhaseA)
        self.assertEqual(captured["task_items"], [task])
        self.assertIs(captured["phase_a"].eval_fn, task["eval_fn"])


if __name__ == "__main__":
    unittest.main()