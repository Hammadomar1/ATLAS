from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ReflexInput:
    t: int
    frame: Sequence[Sequence[Sequence[float]]]


@dataclass(frozen=True)
class ProtoEvent:
    bearing_index: int
    strength: float
    width: float = 1.0
    velocity: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class ReflexDrive:
    drive: Sequence[float]


@dataclass(frozen=True)
class DNFState:
    u: Sequence[float]
    winner_index: int


@dataclass(frozen=True)
class ReflexOutput:
    t: int
    nominal_action: float
    drive: ReflexDrive
    dnf_state: DNFState
    events: List[ProtoEvent]


@dataclass(frozen=True)
class ObserverOutput:
    memory: Sequence[float]
    novelty: float
    soft_veto: float


@dataclass(frozen=True)
class MetaPacket:
    k: int
    delta_theta: Sequence[float]
    created_at_t: int


@dataclass(frozen=True)
class MetaArrival:
    t: int
    delta_theta: Sequence[float]
    k: int
    created_at_t: int


@dataclass(frozen=True)
class ShieldOutput:
    action: float
    used_fallback: bool
    safe_set_empty: bool


@dataclass(frozen=True)
class SupervisorState:
    mode: str
    meta_ok: bool
    guard_enabled: bool
    meta_age: Optional[int]
    accept_meta: bool


@dataclass(frozen=True)
class Telemetry:
    t: int
    downsampled_frame: Sequence[Sequence[Sequence[float]]]
    nominal_action: float
    pre_action: float
    shielded_action: float
    hazard: float
    kappa: float
    safe_set_size: int
    oscillation: float
    mode: str
    novelty: float
    soft_veto: float
    soft_veto_hold: float
    meta_arrived: bool
    meta_age: Optional[int]


@dataclass(frozen=True)
class MetricsRecord:
    t: int
    reflex_duration_s: float
    hazard: float
    oscillation: float
    shield_used_fallback: bool
    meta_age: Optional[int]
