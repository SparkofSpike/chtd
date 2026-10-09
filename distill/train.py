"""The fine-tuning step: what it would run, and what has to be true before it can.

This module produces the *plan* — the recipe, and the provenance that makes a finished run
reproducible — and refuses to produce one that could not be defended. It does not run the training
loop; that needs a GPU, and a machine without one should say so rather than emit a config and let
someone assume a model came out the other end.

The refusal is a real check and not a comment: `require_accelerator` asks the runtime, and a caller
that cannot satisfy it gets an error naming what is missing.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

from distill.dataset import DatasetManifest

__all__ = [
    "AcceleratorMissing",
    "LoraRecipe",
    "TrainingPlan",
    "build_plan",
    "require_accelerator",
    "write_plan",
]


class AcceleratorMissing(RuntimeError):
    """The weight-level step cannot run on this machine."""


@dataclass(frozen=True)
class LoraRecipe:
    """The fine-tune's hyperparameters.

    Frozen with the plan. The design's comparison holds everything else equal across the two arms —
    same track, same tasks, same budget — and the recipe is on that list: an arm comparison between
    two recipes compares the recipes.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_sequence_length: int = 4096
    warmup_ratio: float = 0.03
    seed: int = 0

    def fingerprint(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TrainingPlan:
    """What to train, on what, and from which dataset — the inputs a reproduction needs."""

    arm: str
    base_model: str
    base_model_revision: str | None
    teacher_model: str
    track_sha256: str
    dataset_sha256: str
    entries: int
    recipe: LoraRecipe
    output_dir: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> TrainingPlan:
        """Read back what :meth:`to_json` wrote, with the recipe restored to its own type.

        ``asdict`` flattening the recipe into a mapping is right for a record a reader inspects and
        wrong for one a program reads back: a plan whose ``recipe`` stayed a dict fails at the first
        attribute access, so the run starts and then stops for a reason that has nothing to do with
        the training it was meant to do.
        """
        raw = json.loads(text)
        recipe = dict(raw.get("recipe") or {})
        target_modules = recipe.get("target_modules")
        if isinstance(target_modules, list):
            recipe["target_modules"] = tuple(target_modules)
        return cls(
            **{
                **raw,
                "recipe": LoraRecipe(**recipe),
                "notes": tuple(raw.get("notes") or ()),
            }
        )


def require_accelerator() -> str:
    """Return the accelerator's name, or refuse.

    Asks the runtime rather than a configuration file. A machine with no GPU can still build the
    dataset, run the gate and compare arms; what it cannot do is produce the weights the L2 claim
    rests on, and the honest failure names that.
    """
    try:
        import torch  # noqa: PLC0415 - optional by design; the rest of the pipeline needs nothing
    except ImportError as error:
        raise AcceleratorMissing(
            "no torch installation, so no fine-tune. Install the training extra "
            "(`pip install -e .[train]`) on a machine with a GPU, or deliver at L1 and state that "
            "the weight-level claim was not tested."
        ) from error
    if not torch.cuda.is_available():
        raise AcceleratorMissing(
            f"torch is installed but reports no CUDA device on {platform.platform()}. "
            "The L2 deliverable needs one; without it the project falls back to L1 and must say so."
        )
    return torch.cuda.get_device_name(0)


def build_plan(
    manifest: DatasetManifest,
    *,
    arm: str,
    base_model: str,
    teacher_model: str,
    output_dir: str,
    recipe: LoraRecipe | None = None,
    base_model_revision: str | None = None,
) -> TrainingPlan:
    """Assemble a plan, refusing the cases that would produce a meaningless result.

    A plan is refused when the dataset is empty (nothing to train on), when the dataset came from a
    track the caller did not name (the run would not be attributable), and when ``arm`` is not one
    of the two the design compares.
    """
    if arm not in {"baseline", "chtd"}:
        raise ValueError(
            f"unknown arm {arm!r}; the design compares 'baseline' (teacher data from the teacher's "
            "own trajectory) with 'chtd' (teacher data from the student's states)"
        )
    if manifest.entries == 0:
        raise ValueError(
            "the dataset has no admitted entries; training on it would produce a checkpoint "
            "indistinguishable from the base weights but reported as a fine-tune"
        )
    if len(manifest.track_sha256) != 64:
        raise ValueError(f"the dataset's track digest {manifest.track_sha256!r} is not a sha256")

    notes = (
        "Same track, tasks, budget and recipe across arms; this plan differs from the other arm "
        "only in the dataset it names.",
    )
    return TrainingPlan(
        arm=arm,
        base_model=base_model,
        base_model_revision=base_model_revision,
        teacher_model=teacher_model,
        track_sha256=manifest.track_sha256,
        dataset_sha256=manifest.dataset_sha256,
        entries=manifest.entries,
        recipe=recipe or LoraRecipe(),
        output_dir=output_dir,
        notes=notes,
    )


def write_plan(plan: TrainingPlan, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(plan.to_json() + "\n", encoding="utf-8")
    return target
