"""Generate Factor subclass .py files for surviving signals."""

from pathlib import Path

from src.mining.enumeration import CandidateSpec
from src.mining.evaluate import EvalResult
from src.mining.transforms import TransformType


def generate_factor_file(
    spec: CandidateSpec,
    metrics: EvalResult,
    output_dir: Path,
) -> Path:
    """Generate a Factor subclass .py file for a surviving signal."""
    class_name = _to_class_name(spec.name)
    compute_body = _generate_compute_body(spec)
    formula = _describe_formula(spec)

    code = f'''"""Auto-generated signal: {spec.name}"""

import pandas as pd

from src.data.cleaner import remove_infinities
from src.factors.base import Factor


class {class_name}(Factor):
    """{formula}

    Mining discovery metrics (dev period 1975-2004):
        ICIR={metrics.dev_icir:.3f}, Hit Rate={metrics.dev_hit_rate:.3f},
        Turnover={metrics.turnover:.3f}, Spread t={metrics.dev_spread_t:.2f}
    Validation ICIR (2005-2014): {metrics.val_icir:.3f}
    """

    name = "{spec.name}"
    category = "{spec.category}"

    def compute(self, panel) -> pd.DataFrame:
{compute_body}
'''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{spec.name}.py"
    filepath.write_text(code)
    return filepath


def _to_class_name(name: str) -> str:
    """Convert signal name to PascalCase class name."""
    parts = name.replace(".", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _generate_compute_body(spec: CandidateSpec) -> str:
    """Generate the compute() method body."""
    t = spec.transform
    indent = "        "

    if t == TransformType.LEVEL:
        return f'{indent}return panel.pivot("{spec.field_a}")'

    elif t == TransformType.RATIO_TO_MKTCAP:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}mktcap = panel.pivot("prccm") * panel.pivot("cshoq")\n'
                f'{indent}return remove_infinities(field / mktcap)')

    elif t == TransformType.RATIO_TO_ASSETS:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}atq = panel.pivot("atq")\n'
                f'{indent}return remove_infinities(field / atq)')

    elif t == TransformType.RATIO_TO_EQUITY:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}ceqq = panel.pivot("ceqq")\n'
                f'{indent}return remove_infinities(field / ceqq)')

    elif t == TransformType.RATIO_TO_SALES:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}saleq = panel.pivot("saleq")\n'
                f'{indent}return remove_infinities(field / saleq)')

    elif t == TransformType.GROWTH_YOY:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}return remove_infinities(field / field.shift(12) - 1)')

    elif t == TransformType.GROWTH_QOQ:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}return remove_infinities(field / field.shift(3) - 1)')

    elif t == TransformType.ACCELERATION:
        return (f'{indent}field = panel.pivot("{spec.field_a}")\n'
                f'{indent}growth = remove_infinities(field / field.shift(12) - 1)\n'
                f'{indent}return growth - growth.shift(12)')

    elif t == TransformType.MOMENTUM:
        w = spec.window or 12
        return (f'{indent}trt1m = panel.pivot("trt1m")\n'
                f'{indent}return trt1m.shift(1).rolling(window={w}).mean()')

    elif t == TransformType.VOLATILITY:
        w = spec.window or 12
        return (f'{indent}trt1m = panel.pivot("trt1m")\n'
                f'{indent}return -trt1m.rolling(window={w}).std()')

    elif t == TransformType.HIGH_LOW_RANGE:
        return (f'{indent}prchm = panel.pivot("prchm")\n'
                f'{indent}prclm = panel.pivot("prclm")\n'
                f'{indent}prccm = panel.pivot("prccm")\n'
                f'{indent}return remove_infinities((prchm - prclm) / prccm)')

    elif t == TransformType.TWO_FIELD_RATIO:
        return (f'{indent}a = panel.pivot("{spec.field_a}")\n'
                f'{indent}b = panel.pivot("{spec.field_b}")\n'
                f'{indent}return remove_infinities(a / b)')

    elif t == TransformType.ANALYST_REVISION:
        return (f'{indent}numup = panel.pivot("NUMUP")\n'
                f'{indent}numdown = panel.pivot("NUMDOWN")\n'
                f'{indent}numest = panel.pivot("NUMEST")\n'
                f'{indent}return remove_infinities((numup - numdown) / numest)')

    elif t == TransformType.ANALYST_SUE:
        return (f'{indent}surpmean = panel.pivot("surpmean")\n'
                f'{indent}surpstdev = panel.pivot("surpstdev")\n'
                f'{indent}return remove_infinities(surpmean / surpstdev)')

    elif t == TransformType.ANALYST_DISPERSION:
        return (f'{indent}surpstdev = panel.pivot("surpstdev")\n'
                f'{indent}fy1 = panel.pivot("FY_1")\n'
                f'{indent}return remove_infinities(-surpstdev / fy1.abs())')

    return f'{indent}raise NotImplementedError("Unknown transform")'


def _describe_formula(spec: CandidateSpec) -> str:
    """Human-readable formula description."""
    t = spec.transform
    a, b = spec.field_a, spec.field_b

    formulas = {
        TransformType.LEVEL: f"Raw {a}",
        TransformType.RATIO_TO_MKTCAP: f"{a} / market_cap",
        TransformType.RATIO_TO_ASSETS: f"{a} / total_assets",
        TransformType.RATIO_TO_EQUITY: f"{a} / common_equity",
        TransformType.RATIO_TO_SALES: f"{a} / sales",
        TransformType.GROWTH_YOY: f"YoY growth of {a}",
        TransformType.GROWTH_QOQ: f"QoQ growth of {a}",
        TransformType.ACCELERATION: f"YoY acceleration of {a}",
        TransformType.MOMENTUM: f"{spec.window}M momentum",
        TransformType.VOLATILITY: f"Negative {spec.window}M volatility",
        TransformType.HIGH_LOW_RANGE: f"(high - low) / close",
        TransformType.TWO_FIELD_RATIO: f"{a} / {b}",
        TransformType.ANALYST_REVISION: f"(NUMUP - NUMDOWN) / NUMEST",
        TransformType.ANALYST_SUE: f"surpmean / surpstdev (SUE)",
        TransformType.ANALYST_DISPERSION: f"-surpstdev / |FY_1| (negative dispersion)",
    }
    return formulas.get(t, f"{a} transform {t.value}")
