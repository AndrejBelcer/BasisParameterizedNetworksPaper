"""Optional CPU second-moment normalization of current flow-matching gradients."""
import math

import torch

from .parameter_space import require_finite, vector_norm


class GradientSecondMoment:
    def __init__(self, decay=.9, damping=.05):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("second_moment_decay must be in [0, 1)")
        if not math.isfinite(damping) or not 0 < damping <= 1:
            raise ValueError("second_moment_damping must be in (0, 1]")
        self.decay, self.damping = decay, damping
        self.second = None
        self.updates = 0

    def direction(self, directions, norms):
        """Update once per new RGB-guide collection, never per KID probe.

        Guides are unit vectors; restore their RAW norms before squaring.
        The numerator is today's mean, not a momentum estimate. This state is
        local to this invocation; loading a model starts a fresh history.
        """
        if not directions or len(directions) != len(norms):
            raise ValueError("Second moments need matching guide directions and norms")
        mean = torch.zeros_like(directions[0], device="cpu", dtype=torch.float32)
        squares = torch.zeros_like(mean)
        for direction, norm in zip(directions, norms):
            if not math.isfinite(norm) or norm < 0:
                raise FloatingPointError("Invalid RGB norm in second-moment update")
            raw = direction.detach().cpu().float()*norm
            require_finite(raw, "raw RGB gradient")
            mean.add_(raw, alpha=1/len(directions))
            squares.addcmul_(raw, raw, value=1/len(directions))
        if self.second is None:
            self.second = torch.zeros_like(mean)
        if self.second.shape != mean.shape:
            raise ValueError("Model shape changed during the second-moment history")
        updated = self.decay*self.second+(1-self.decay)*squares
        require_finite(updated, "running gradient second moment")
        self.second = updated
        self.updates += 1
        rms = (self.second/(1-self.decay**self.updates)).sqrt()
        global_rms = vector_norm(rms)/math.sqrt(rms.numel())
        floor = max(global_rms*self.damping, torch.finfo(torch.float32).tiny)
        direction = -mean/rms.clamp_min(floor)
        require_finite(direction, "preconditioned RGB direction")
        length = vector_norm(direction)
        if length:
            direction.div_(length)
        return direction, dict(updates=self.updates, decay=self.decay, damping=self.damping,
                               global_rms=global_rms, denominator_floor=floor,
                               floored_fraction=float((rms < floor).float().mean()))
