"""Paper benchmark registry."""

from .engineering import (
    CarCabDesign,
    CarSideImpact,
    ConceptualMarine,
    DiscBrake,
    FourBarTruss,
    GearTrain,
    HatchCover,
    PressureVessel,
    RocketInjector,
    SpeedReducer,
    WaterResource,
)
from .extensions import WFG4, WeierstrassTakagi
from .hpo import make_lego, make_materials, make_mic, make_ship
from .synthetic import make_abc, make_abd, make_acd, make_ar, make_bc, make_bcd


_BENCHMARKS = {
    "bc": make_bc,
    "ar": make_ar,
    "abc": make_abc,
    "abd": make_abd,
    "acd": make_acd,
    "bcd": make_bcd,
    "re21": FourBarTruss,
    "re23": PressureVessel,
    "re24": HatchCover,
    "re33": DiscBrake,
    "re35": SpeedReducer,
    "re36": GearTrain,
    "re37": RocketInjector,
    "re41": CarSideImpact,
    "re42": ConceptualMarine,
    "re61": WaterResource,
    "re91": CarCabDesign,
    "lego": make_lego,
    "materials": make_materials,
    "mic": make_mic,
    "ship": make_ship,
    "wfg4_30d": lambda: WFG4(n_dim=30),
    "wfg4_40d": lambda: WFG4(n_dim=40),
    "weierstrass_takagi": WeierstrassTakagi,
}

_ALIASES = {
    "four_bar_truss": "re21",
    "four_bar_truss_design": "re21",
    "pressure_vessel": "re23",
    "pressure_vessel_design": "re23",
    "hatch_cover": "re24",
    "hatch_cover_design": "re24",
    "disc_brake": "re33",
    "disk_brake": "re33",
    "speed_reducer": "re35",
    "speed_reducer_design": "re35",
    "gear_train": "re36",
    "gear_train_design": "re36",
    "rocket_injector": "re37",
    "rocket_injector_design": "re37",
    "car_side_impact": "re41",
    "car_side_impact_design": "re41",
    "conceptual_marine": "re42",
    "conceptual_marine_design": "re42",
    "water_resource": "re61",
    "water_resource_planning": "re61",
    "car_cab": "re91",
    "car_cab_design": "re91",
    "mat": "materials",
    "wtmop": "weierstrass_takagi",
}


def _normalize_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def get_benchmark(name):
    """Create a benchmark by paper abbreviation or descriptive alias."""
    key = _normalize_name(name)
    key = _ALIASES.get(key, key)
    try:
        factory = _BENCHMARKS[key]
    except KeyError as error:
        available = ", ".join(_BENCHMARKS)
        raise ValueError(
            f"Unknown benchmark {name!r}. Available benchmarks: {available}."
        ) from error
    return factory()


def list_benchmarks():
    """Return canonical benchmark identifiers in paper order."""
    return tuple(_BENCHMARKS)


__all__ = ["get_benchmark", "list_benchmarks"]
