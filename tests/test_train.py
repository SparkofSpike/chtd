"""The training step's job here is to refuse honestly and to record what a run would need."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from distill.dataset import DatasetManifest
from distill.train import (
    AcceleratorMissing,
    LoraRecipe,
    TrainingPlan,
    build_plan,
    require_accelerator,
    write_plan,
)

TRACK = "a" * 64


def _manifest(entries: int = 12, track: str = TRACK) -> DatasetManifest:
    return DatasetManifest(
        track_sha256=track,
        dataset_sha256="b" * 64,
        entries=entries,
        refusals=3,
        tasks=("t1", "t2"),
        arms=("baseline", "chtd"),
        created_at="2026-10-09T00:00:00Z",
    )


def test_the_accelerator_is_asked_for_rather_than_assumed() -> None:
    # Asks the runtime, not a config file. This machine has no GPU, so the honest answer is a
    # refusal that names what is missing — and that is what should happen here.
    with pytest.raises(AcceleratorMissing) as caught:
        require_accelerator()
    message = str(caught.value)
    assert "GPU" in message or "CUDA" in message or "torch" in message
    # The fallback the refusal points at is the one the design names.
    assert "L1" in message or "train" in message


def test_an_empty_dataset_is_refused() -> None:
    # Training on it would produce a checkpoint indistinguishable from the base weights but
    # reported as a fine-tune.
    with pytest.raises(ValueError, match="no admitted entries"):
        build_plan(_manifest(entries=0), arm="chtd", base_model="qwen", teacher_model="qwen-large", output_dir="/out")


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_plan(_manifest(), arm="treatment", base_model="qwen", teacher_model="qwen-large", output_dir="/out")


def test_a_short_track_digest_is_refused() -> None:
    with pytest.raises(ValueError, match="not a sha256"):
        build_plan(
            _manifest(track="short"), arm="chtd", base_model="qwen", teacher_model="qwen-large", output_dir="/out"
        )


def test_a_plan_records_the_provenance_a_reproduction_needs() -> None:
    plan = build_plan(
        _manifest(),
        arm="chtd",
        base_model="Qwen/Qwen3-4B",
        base_model_revision="abc123",
        teacher_model="Qwen/Qwen3-32B",
        output_dir="/out/chtd",
    )
    assert plan.track_sha256 == TRACK
    assert plan.dataset_sha256 == "b" * 64
    assert plan.base_model_revision == "abc123"
    assert plan.entries == 12
    payload = json.loads(plan.to_json())
    assert payload["recipe"]["rank"] == 16
    assert payload["arm"] == "chtd"


def test_the_plan_says_the_arms_differ_only_in_their_dataset() -> None:
    plan = build_plan(
        _manifest(), arm="baseline", base_model="qwen", teacher_model="qwen-large", output_dir="/out"
    )
    assert any("only in the dataset" in note for note in plan.notes)


def test_the_recipe_is_frozen_with_the_plan() -> None:
    # An arm comparison between two recipes compares the recipes.
    first = build_plan(
        _manifest(), arm="chtd", base_model="qwen", teacher_model="qwen-large", output_dir="/out"
    )
    second = build_plan(
        _manifest(),
        arm="chtd",
        base_model="qwen",
        teacher_model="qwen-large",
        output_dir="/out",
        recipe=LoraRecipe(rank=64),
    )
    assert first.recipe.fingerprint() != second.recipe.fingerprint()
    assert first.recipe.fingerprint() == LoraRecipe().fingerprint()


def test_a_plan_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    plan = build_plan(
        _manifest(), arm="chtd", base_model="qwen", teacher_model="qwen-large", output_dir="/out"
    )
    path = write_plan(plan, tmp_path / "nested" / "plan.json")
    assert json.loads(path.read_text(encoding="utf-8"))["dataset_sha256"] == "b" * 64


def test_a_plan_reads_back_equal_to_the_one_that_was_written(tmp_path: Path) -> None:
    """`to_json` flattens the recipe with `asdict`; `from_json` has to put it back.

    A plan whose `recipe` stayed a mapping fails at the first attribute access, so the run starts
    and then stops for a reason unrelated to the training it was meant to do. Equality on the whole
    object is the assertion that catches it, because the recipe is the field that does not survive
    a naive round trip.
    """
    plan = build_plan(
        _manifest(),
        arm="chtd",
        base_model="qwen",
        teacher_model="qwen-large",
        output_dir="/out",
        base_model_revision="abc123",
    )
    path = write_plan(plan, tmp_path / "plan.json")
    restored = TrainingPlan.from_json(path.read_text(encoding="utf-8"))
    assert restored == plan
    assert isinstance(restored.recipe, LoraRecipe)
    assert restored.recipe.target_modules == plan.recipe.target_modules


def test_the_default_recipe_is_one_choice_and_not_a_guess() -> None:
    recipe = LoraRecipe()
    assert recipe.rank > 0 and recipe.alpha > 0
    assert 0.0 <= recipe.dropout < 1.0
    assert recipe.target_modules
    assert recipe.max_sequence_length > 0
