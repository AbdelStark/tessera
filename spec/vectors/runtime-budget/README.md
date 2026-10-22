# Runtime budget vectors

[`generate.py`](generate.py) independently counts the parent rows, raw final
evaluations and batch instances at every shared tree level, from literal leaf
populations under RFC-0010. [`geometry.json`](geometry.json) covers empty and
height-zero trees, fraction and product trees, delayed joins and forests whose
root variables exceed the tallest fraction tree's height.

The runtime budget additionally uses the independently derived mixed stage
schedule, complete canonical opening groups and actual matrix layouts
([RFC-0014](../../../docs/rfcs/RFC-0014-interaction-and-soundness.md#soundness-calculator),
[RFC-0053](../../../docs/rfcs/RFC-0053-greedy-runtime-boundary.md#schedule-transcript-and-budget)).
Actual CPU runtime proofs compare their message lengths against this derivation;
these geometry fixtures alone certify no model or parameter set.
