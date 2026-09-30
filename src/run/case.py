# case description for the generic runner: every setting that decides the numbers of a run
from dataclasses import dataclass, field
import numpy as np

minimum_relaxation_time = 0.501 # regularized floor without LES (DESIGN 7a, E2)

def check_choice(name, value, choices):
    """
    Raise if a string setting is not one of its allowed values.
    """

    if value not in choices:
        raise ValueError(f"{name} = {value!r}, expected one of {choices}")
    
@dataclass
class Flow:
    """
    Free stream, Reynolds number on a reference length, optional body force; viscosity and tau follow.
    """

    free_stream_velocity: float = 0.05 # U, lattice units
    reynolds_number: float = 5000.0 # Re on reference_length, unused when relaxation_time_override is set
    reference_length: float = 80.0 # L, cells
    relaxation_time_override: float | None = None # tau given directly (channel, shear wave)
    body_force_x: float = 0.0 # g, Guo forcing

    @property
    def viscosity(self):
        """
        Returns nu = U L / Re, or (tau - 1/2) / 3 when tau is given.
        """

        if self.relaxation_time_override is not None:
            return (self.relaxation_time_override - 0.5) / 3

        return self.free_stream_velocity * self.reference_length / self.reynolds_number

    @property
    def relaxation_time(self):
        """
        Returns tau = 3 nu + 1/2, or the given tau.
        """

        if self.relaxation_time_override is not None:
            return self.relaxation_time_override
        
        return 3 * self.viscosity + 0.5

@dataclass
class Domain:
    """
    Grid size, boundaries and relaxation layers.
    """

    nx: int = 800
    ny: int = 400
    nz: int = 4

    x_boundary: str = "inflow" # "inflow" or "periodic"
    y_boundary: str = "walls" # "walls" or "free_slip" or "periodic"

    floor: str = "static" # y walls only: "static" or "moving" (at U)
    ceiling: str = "static" # y walls only: "static" or "moving" (at U)
    side_walls: str = "periodic" # z: "periodic" or "free_slip"

    lid_velocity: float = 0.0 # cavity lid, 2D engine port

    relax_width_x: int = 24 # 0 disables the x layers
    relax_width_z: int = 0 # 0 disables the z layers
    relax_sigma: float = 0.1
    relax_mean_rate: float = 0.0 # z-layer running-mean rate

    layer_kind: str = "fused" # "fused" (inside collide_reg) or "separate" (sponge_relax)

@dataclass
class Turbulence:
    """
    Subgrid model and wall model.
    """

    sgs: str = "wale" # "none" or "wale" or "smagorinsky"
    wale_constant: float = 0.5 # c_w
    smagorinsky_constant: float = 0.04 # c_s: the floor under WALE, the full constant for "smagorinsky"
    wall_model: bool = False # log-law, staircase walls only

@dataclass
class Timing:
    """
    Run length in flow-throughs (nx / U); overrides reproduce examples counted in other units.
    """

    ramp_flow_throughs: float = 1.0
    warmup_flow_throughs: float = 4.0
    total_flow_throughs: float = 10.0
    steps_override: int | None = None
    warmup_override: int | None = None
    sample_every: int = 25
    sample_window: str = "series" # "series" (every sample_every after warmup) or "end" (last step only)

@dataclass
class Part:
    """
    One body with its own force; part id = position in Case.parts + 1.
    """

    name: str
    solid: np.ndarray # (nx, ny, nz) int32 mask
    reference_area: float # A for the coefficients, cells^2
    signed_distance: np.ndarray | None = None # phi: given -> Bouzidi walls, None -> staircase
    wall_velocity: np.ndarray | None = None # (3, nx, ny, nz) rigid surface velocity (spin), None -> at rest

@dataclass
class Case:
    """
    Everything the runner needs for one run.
    """

    name: str
    tag: str # output file suffix
    flow: Flow
    domain: Domain
    turbulence: Turbulence
    timing: Timing

    parts: list[Part] = field(default_factory=list)
    lattice: str = "D3Q19"

    dimensions: int = 3
    collision: str = "regularized" # "bgk" or "trt" or "regularized"
    inlet: str = "neem_open" # "neem_open" or "neem" or "none"
    start: str = "rest_ramp" # "rest_ramp" or "uniform" or "custom"

    initial_velocity: np.ndarray | None = None # (3, nx, ny, nz), start = "custom" only
    allow_below_floor: bool = False # tau < 0.501 on purpose (LES carries stability); logged

    def flow_through_steps(self):
        """
        Returns T_ft = nx / U in steps.
        """

        return self.domain.nx / self.flow.free_stream_velocity
    
    def total_steps(self):
        """
        Returns the run length in steps.
        """

        if self.timing.steps_override is not None:
            return self.timing.steps_override
        
        return round(self.flow_through_steps() * self.timing.total_flow_throughs)
    
    def warmup_steps(self):
        """
        Returns the steps discarded before averaging.
        """

        if self.timing.warmup_override is not None:
            return self.timing.warmup_override
        
        return round(self.flow_through_steps() * self.timing.warmup_flow_throughs)
    
    def ramp_steps(self):
        """
        Returns the cosine inlet ramp length in steps.
        """

        return round(self.flow_through_steps() * self.timing.ramp_flow_throughs)
    
    def validate(self):
        """
        Reject settings the solver cannot run or that contradict each other.
        """

        check_choice("lattice", self.lattice, ("D3Q19",))
        check_choice("dimensions", self.dimensions, (3,))
        check_choice("collision", self.collision, ("bgk", "trt", "regularized"))
        check_choice("inlet", self.inlet, ("neem_open", "neem", "none"))
        check_choice("start", self.start, ("rest_ramp", "uniform", "custom"))
        check_choice("x_boundary", self.domain.x_boundary, ("inflow", "periodic"))
        check_choice("y_boundary", self.domain.y_boundary, ("walls", "free_slip", "periodic"))
        check_choice("floor", self.domain.floor, ("static", "moving"))
        check_choice("ceiling", self.domain.ceiling, ("static", "moving"))
        check_choice("side_walls", self.domain.side_walls, ("periodic", "free_slip"))
        check_choice("layer_kind", self.domain.layer_kind, ("fused", "separate"))
        check_choice("sgs", self.turbulence.sgs, ("none", "wale", "smagorinsky"))
        check_choice("sample_window", self.timing.sample_window, ("series", "end"))

        # physics limits
        if self.flow.relaxation_time < minimum_relaxation_time and not self.allow_below_floor:
            raise ValueError(f"tau {self.flow.relaxation_time:.5f} below {minimum_relaxation_time}: raise L or U, lower Re, or set allow_below_floor")

        # combinations the runner does not support
        if (self.domain.x_boundary == "periodic") != (self.inlet == "none"):
            raise ValueError("periodic x needs inlet = 'none', and inlet = 'none' needs periodic x")
        if (self.start == "custom") != (self.initial_velocity is not None):
            raise ValueError("start = 'custom' needs initial_velocity, and initial_velocity needs start = 'custom'")
        if self.domain.layer_kind == "fused" and self.collision != "regularized":
            raise ValueError("fused layers live in collide_reg: use layer_kind = 'separate' for bgk / trt")
        if self.domain.lid_velocity != 0.0:
            raise ValueError("lid_velocity is reserved for the 2D engine port")
        
        # parts
        grid_shape = (self.domain.nx, self.domain.ny, self.domain.nz)
        for part in self.parts:
            if part.solid.shape != grid_shape:
                raise ValueError(f"part {part.name} mask {part.solid.shape} != grid {grid_shape}")
        if len({part.signed_distance is None for part in self.parts}) > 1:
            raise ValueError("mixed Bouzidi and staircase parts are not supported yet")
        if self.turbulence.wall_model and any(part.signed_distance is not None for part in self.parts):
            raise ValueError("wall model needs staircase walls")