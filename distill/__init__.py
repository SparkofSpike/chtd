"""Build the training dataset, and refuse to build a dishonest one.

The design's pipeline is: the student runs, the teacher answers at the student's states, and where
the two differ the teacher's decision becomes a label. This module is the step that turns those
answers into rows, and it holds two gates the rest of the pipeline depends on:

* **The co-harness gate.** Two runs may only be compared if they ran the same track. The check is a
  recorded track digest and not a claim, because a dataset built from two different recipes is a
  measurement of two different experiments wearing one name.
* **The signal-quality gate.** A divergence is not automatically a place where the teacher was
  right. Only divergences whose teacher-side choice was validated by the outcome are admitted.

Both gates refuse by raising. A pipeline that quietly dropped what it could not vouch for would
produce a dataset whose provenance nobody could reconstruct, which is the failure the write-up is
supposed to make impossible.
"""

from __future__ import annotations

__all__: list[str] = []
