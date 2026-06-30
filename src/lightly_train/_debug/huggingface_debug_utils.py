#
# #
# # Copyright 2020 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.##
"""Minimal subset of ``transformers.debug_utils`` used by LightlyTrain.

LightlyTrain vendors only the parts of HuggingFace's debug utility that the
:class:`UnderflowOverflowMonitor` actually depends on: the
:class:`DebugUnderflowOverflow` class plus its ``get_abs_min_max`` and
``detect_overflow`` module-level helpers. The upstream module additionally
defines a ``DebugOption`` enum that LightlyTrain does not use.

Modifications Copyright 2026 Lightly AG:

  - Replaced the conditional ``torch`` import (``if is_torch_available(): import
    torch``) with an unconditional ``import torch`` (torch is always available
    inside LightlyTrain).
  - Removed the ``transformers.utils`` dependency (``ExplicitEnum``,
    ``is_torch_available``, the package-local ``logging``-style ``get_logger``);
    use the stdlib ``logging`` module instead.
  - Added explicit type annotations to constructor parameters, methods, and locals
    so the file satisfies the repo's strict mypy configuration without ``# type:
    ignore`` comments.
  - Dropped the unused ``DebugOption`` enum.
  - Added optional ``log_file`` constructor argument: when provided, all
    ``print()``-based output (frame dumps, "Detected inf/nan" headers, and the
    overflow/nan notices from :func:`detect_overflow`) is redirected to the
    file and flushed after every forward. Upstream always writes to ``stdout``.
  - Added optional ``batch_number_mode="auto"|"manual"`` constructor argument.
    ``"auto"`` preserves upstream behavior (``batch_number`` is incremented by
    the root module's forward hook). ``"manual"`` disables that auto-increment
    and lets callers drive ``batch_number`` explicitly via :meth:`set_batch_number`
    (used by training loops that never call the root module's ``forward``).
  - In both modes the "Starting batch number=..." header is emitted on advance
    rather than only at the first forward of training; this is a strict
    superset of upstream and makes manual-mode output deterministic.
  - Captured the per-module :class:`RemovableHandle` returned by
    ``Module.register_forward_hook`` and added :meth:`detach_hooks` so callers
    can remove only the hooks this monitor added without disturbing any
    pre-existing user forward hooks.
  - Wrapped ``analyse_model``/``register_forward_hook`` in ``__init__`` with a
    ``try``/``except`` that calls :meth:`detach_hooks` on partial-failure
    cleanup (upstream leaks registered hooks on construction failure).
"""

from __future__ import annotations

import collections
import contextlib
import logging
from typing import IO, Any, Literal

import torch
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


class DebugUnderflowOverflow:
    """
    This debug class helps detect and understand where the model starts getting very large or very small, and more
    importantly `nan` or `inf` weight and activation elements.

    There are 2 working modes:

    1. Underflow/overflow detection (default)
    2. Specific batch absolute min/max tracing without detection

    Mode 1: Underflow/overflow detection

    To activate the underflow/overflow detection, initialize the object with the model :

    ```python
    debug_overflow = DebugUnderflowOverflow(model)
    ```

    then run the training as normal and if `nan` or `inf` gets detected in at least one of the weight, input or output
    elements this module will throw an exception and will print `max_frames_to_save` frames that lead to this event,
    each frame reporting

    1. the fully qualified module name plus the class name whose `forward` was run
    2. the absolute min and max value of all elements for each module weights, and the inputs and output

    For example, here is the header and the last few frames in detection report for `google/mt5-small` run in fp16
    mixed precision :

    ```
    Detected inf/nan during batch_number=0
    Last 21 forward frames:
    abs min  abs max  metadata
    [...]
                      encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
    2.17e-07 4.50e+00 weight
    1.79e-06 4.65e+00 input[0]
    2.68e-06 3.70e+01 output
                      encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
    8.08e-07 2.66e+01 weight
    1.79e-06 4.65e+00 input[0]
    1.27e-04 2.37e+02 output
                      encoder.block.2.layer.1.DenseReluDense.wo Linear
    1.01e-06 6.44e+00 weight
    0.00e+00 9.74e+03 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
    1.79e-06 4.65e+00 input[0]
    3.18e-04 6.27e+04 output
                      encoder.block.2.layer.1.dropout Dropout
    3.18e-04 6.27e+04 input[0]
    0.00e+00      inf output
    ```

    You can see here, that `T5DenseGatedGeluDense.forward` resulted in output activations, whose absolute max value was
    around 62.7K, which is very close to fp16's top limit of 64K. In the next frame we have `Dropout` which
    renormalizes the weights, after it zeroed some elements, which pushes the absolute max value to more than
    64K, and we get an overflow.

    As you can see it's the previous frames that we need to look into when the numbers start going into very large for
    fp16 numbers.

    The tracking is done in a forward hook, which gets invoked immediately after `forward` has completed.

    By default the last 21 frames are printed. You can change the default to adjust for your needs. For example :

    ```python
    debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
    ```

        To validate that you have set up this debugging feature correctly, and you intend to use it in a training that
        may take hours to complete, first run it with normal tracing enabled for one of a few batches as explained in
        the next section.


        Mode 2. Specific batch absolute min/max tracing without detection

        The second work mode is per-batch tracing with the underflow/overflow detection feature turned off.

        Let's say you want to watch the absolute min and max values for all the ingredients of each `forward` call of a
    given batch, and only do that for batches 1 and 3. Then you instantiate this class as :

    ```python
    debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
    ```

    And now full batches 1 and 3 will be traced using the same format as explained above. Batches are 0-indexed.

    This is helpful if you know that the program starts misbehaving after a certain batch number, so you can
    fast-forward right to that area.


    Early stopping:

    You can also specify the batch number after which to stop the training, with :

    ```python
    debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
    ```

    This feature is mainly useful in the tracing mode, but you can use it for any mode.


    **Performance**:

    As this module measures absolute `min`/``max` of each weight of the model on every forward it'll slow the training
    down. Therefore remember to turn it off once the debugging needs have been met.

    Args:
        model (`nn.Module`):
            The model to debug.
        max_frames_to_save (`int`, *optional*, defaults to 21):
            How many frames back to record
        trace_batch_nums(`list[int]`, *optional*, defaults to `[]`):
            Which batch numbers to trace (turns detection off)
        abort_after_batch_num  (`int``, *optional*):
            Whether to abort after a certain batch number has finished
        log_file (`IO[str]`, *optional*):
            File-like object to redirect all ``print()`` output to. When ``None``
            (the default) output goes to ``stdout`` (upstream behavior). When
            set, the file is flushed after every forward.
        batch_number_mode (`"auto"|"manual"`, *optional*, defaults to `"auto"`):
            How ``batch_number`` advances. ``"auto"`` (upstream behavior) lets
            the root module's forward hook increment it. ``"manual"`` disables
            that auto-increment; callers must drive ``batch_number`` via
            :meth:`set_batch_number` (used by training loops that never call
            the root module's ``forward``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_frames_to_save: int = 21,
        trace_batch_nums: list[int] | None = None,
        abort_after_batch_num: int | None = None,
        log_file: IO[str] | None = None,
        batch_number_mode: Literal["auto", "manual"] = "auto",
    ) -> None:
        if trace_batch_nums is None:
            trace_batch_nums = []
        if batch_number_mode not in ("auto", "manual"):
            raise ValueError(
                f"batch_number_mode must be 'auto' or 'manual', got {batch_number_mode!r}."
            )
        self.model = model
        self.trace_batch_nums = trace_batch_nums
        self.abort_after_batch_num = abort_after_batch_num
        self.log_file = log_file
        self.batch_number_mode = batch_number_mode

        # keep a LIFO buffer of frames to dump as soon as inf/nan is encountered to give context to the problem emergence
        self.frames: collections.deque[str] = collections.deque([], max_frames_to_save)
        self.frame: list[str] = []
        self.batch_number = 0
        self.total_calls = 0
        self.detected_overflow = False
        self.prefix = "                 "
        # ``RemovableHandle``s returned by the per-module
        # ``register_forward_hook`` calls (issued via ``model.apply``) so
        # :meth:`detach_hooks` removes only the hooks this monitor added,
        # leaving any pre-existing forward hooks (user instrumentation, etc.)
        # intact.
        self._handles: list[RemovableHandle] = []
        # Sentinel used by ``forward_hook`` to detect when ``batch_number`` has
        # advanced and a new "Starting batch" header should be emitted.
        # Initialized to -1 so the first forward always emits a header.
        self._prev_batch_number = -1

        try:
            self.analyse_model()
            self.register_forward_hook()
        except BaseException:
            # ``register_forward_hook`` registers via ``model.apply`` and
            # ``analyse_model`` can raise on malformed modules; if registration
            # got partway, remove the handles we captured so far before letting
            # the exception propagate.
            self.detach_hooks()
            raise

    def save_frame(self, frame: str | None = None) -> None:
        if frame is not None:
            self.expand_frame(frame)
        self.frames.append("\n".join(self.frame))
        self.frame = []  # start a new frame

    def expand_frame(self, line: str) -> None:
        self.frame.append(line)

    def trace_frames(self) -> None:
        print("\n".join(self.frames))
        self.frames.clear()

    def reset_saved_frames(self) -> None:
        self.frames.clear()

    def dump_saved_frames(self) -> None:
        print(f"\nDetected inf/nan during batch_number={self.batch_number}")
        print(f"Last {len(self.frames)} forward frames:")
        print(f"{'abs min':8} {'abs max':8} metadata")
        print("\n".join(self.frames))
        print("\n\n")
        self.frames.clear()

    def analyse_model(self) -> None:
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        self.module_names = {m: name for name, m in self.model.named_modules()}
        # self.longest_module_name = max(len(v) for v in self.module_names.values())

    def analyse_variable(self, var: Any, ctx: str) -> None:
        if torch.is_tensor(var):
            self.expand_frame(get_abs_min_max(var, ctx))
            if detect_overflow(var, ctx):
                self.detected_overflow = True
        elif var is None:
            self.expand_frame(f"{'None':>17} {ctx}")
        else:
            self.expand_frame(f"{'not a tensor':>17} {ctx}")

    def batch_start_frame(self) -> None:
        self.expand_frame(
            f"\n\n{self.prefix} *** Starting batch number={self.batch_number} ***"
        )
        self.expand_frame(f"{'abs min':8} {'abs max':8} metadata")

    def batch_end_frame(self) -> None:
        self.expand_frame(
            f"{self.prefix} *** Finished batch number={self.batch_number - 1} ***\n\n"
        )

    def create_frame(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        self.expand_frame(
            f"{self.prefix} {self.module_names[module]} {module.__class__.__name__}"
        )

        # params
        for name, p in module.named_parameters(recurse=False):
            self.analyse_variable(p, name)

        # inputs
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                self.analyse_variable(x, f"input[{i}]")
        else:
            self.analyse_variable(input, "input")

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        self.save_frame()

    def register_forward_hook(self) -> None:
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module: torch.nn.Module) -> None:
        # Capture the returned handle so :meth:`detach_hooks` can remove only
        # the hooks this monitor added (preserves pre-existing user forward
        # hooks).
        handle = module.register_forward_hook(self.forward_hook)
        self._handles.append(handle)

    def detach_hooks(self) -> None:
        """Remove only the forward hooks registered by this monitor.

        Iterates the captured :class:`RemovableHandle` list (see
        :meth:`_register_forward_hook`). Safe to call when no hooks are
        registered (empty list) or when called more than once.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_batch_number(self, batch_number: int) -> None:
        """Set the current batch number explicitly (manual batch-numbering mode).

        Use this in training loops that never call the root module's ``forward``
        (e.g. ``TrainModel.training_step`` invoking submodules directly) so the
        monitor can still tag frames with the correct batch. Should be called
        before each forward pass. No-op when ``batch_number_mode="auto"`` (the
        root module's forward still drives ``batch_number``).
        """
        self.batch_number = batch_number

    def forward_hook(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors
        if self.log_file is not None:
            cm: Any = contextlib.redirect_stdout(self.log_file)
        else:
            cm = contextlib.nullcontext()
        with cm:
            try:
                # Emit a batch header whenever ``batch_number`` advances. In
                # ``"auto"`` mode the root module's forward below increments
                # ``batch_number`` and the next batch's header is emitted at
                # the start of its first submodule forward (equivalent to
                # upstream's post-root emit, just expressed via
                # header-on-advance). In ``"manual"`` mode the caller drives
                # ``batch_number`` via :meth:`set_batch_number` and the header
                # is emitted on the first forward after each set.
                if self.batch_number != self._prev_batch_number:
                    self.batch_start_frame()
                    self._prev_batch_number = self.batch_number
                self.total_calls += 1

                trace_mode = self.batch_number in self.trace_batch_nums
                if trace_mode:
                    self.reset_saved_frames()

                # count batch numbers - in auto mode the root module's forward
                # increments ``batch_number`` so the just-completed batch ends
                # here; in manual mode this is disabled.
                if self.batch_number_mode == "auto" and module == self.model:
                    self.batch_number += 1

                self.create_frame(module, input, output)

                if trace_mode:
                    self.trace_frames()

                if self.detected_overflow and not trace_mode:
                    self.dump_saved_frames()

                    # now we can abort, as it's pointless to continue running
                    raise ValueError(
                        "DebugUnderflowOverflow: inf/nan detected, aborting as "
                        "there is no point running further. Please check the "
                        "debug log file for the activation values prior to "
                        "this event."
                    )

                # abort after certain batch if requested to do so
                if (
                    self.abort_after_batch_num is not None
                    and self.batch_number > self.abort_after_batch_num
                ):
                    raise ValueError(
                        f"DebugUnderflowOverflow: aborting after batch "
                        f"{self.batch_number} due to "
                        f"`abort_after_batch_num={self.abort_after_batch_num}`."
                    )
            finally:
                # Always flush so partial dumps are recoverable even if we
                # raise before the next explicit flush downstream.
                if self.log_file is not None:
                    self.log_file.flush()


def get_abs_min_max(var: torch.Tensor, ctx: str) -> str:
    abs_var = var.abs()
    return f"{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}"


def detect_overflow(var: torch.Tensor, ctx: str) -> bool:
    """
    Report whether the tensor contains any `nan` or `inf` entries.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the tensor in question.

    This function contains a few other helper features that you can enable and tweak directly if you want to track
    various other things.

    Args:
        var: the tensor variable to check
        ctx: the message to print as a context

    Return:
        `True` if `inf` or `nan` was detected, `False` otherwise
    """
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # if needed to monitor large elements can enable the following
    if 0:  # and detected:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    if 0:
        print(
            f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})"
        )

    return detected
