"""SVM (C, gamma) grids for experiments: CLI parsing, Cartesian products, model expansion."""

from __future__ import annotations

from collections.abc import Iterable

SvmPair = tuple[float, float | str]


def parse_svm_pair_strings(raw: list[str] | None) -> list[SvmPair] | None:
    """Parse ``C,gamma`` strings (gamma may be ``scale``, ``auto``, or float). ``None``/empty → no grid."""
    if not raw:
        return None
    out: list[SvmPair] = []
    for item in raw:
        parts = item.split(",", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --svm-pair {item!r}; use C,gamma e.g. 1,scale or 10,0.01"
            )
        c = float(parts[0].strip())
        gr = parts[1].strip()
        g: float | str = gr if gr in ("scale", "auto") else float(gr)
        out.append((c, g))
    return out


def svm_cartesian_pairs(c_list: list[float], gamma_tokens: list[str]) -> list[SvmPair]:
    """All ``(C, gamma)`` combinations; each gamma token is ``scale``, ``auto``, or a float string."""
    out: list[SvmPair] = []
    for c in c_list:
        for gstr in gamma_tokens:
            gs = gstr.strip()
            out.append((c, gs) if gs in ("scale", "auto") else (c, float(gs)))
    return out


def resolve_svm_pairs_from_cli(
    svm_pair: list[str] | None,
    svm_c_grid: list[float] | None,
    svm_gamma_grid: list[str] | None,
) -> list[SvmPair] | None:
    """Resolve ``--svm-pair`` vs ``--svm-c-grid`` / ``--svm-gamma-grid``. Raises ``ValueError`` on invalid CLI."""
    has_c, has_g = svm_c_grid is not None, svm_gamma_grid is not None
    if has_c != has_g:
        raise ValueError("--svm-c-grid and --svm-gamma-grid must be given together")
    if has_c and has_g and svm_pair:
        raise ValueError(
            "use either --svm-pair or (--svm-c-grid with --svm-gamma-grid), not both"
        )
    if has_c and has_g:
        return svm_cartesian_pairs(svm_c_grid, svm_gamma_grid)
    return parse_svm_pair_strings(svm_pair)


def expand_models_with_svm(
    models: Iterable[str],
    svm_pairs: list[SvmPair] | None,
) -> list[tuple[str, dict[str, float | str]]]:
    """One train job per model; non-empty ``svm_pairs`` expands ``svm`` into one row per ``(C, gamma)``."""
    out: list[tuple[str, dict[str, float | str]]] = []
    for m in models:
        if m == "svm":
            if svm_pairs:
                for c, g in svm_pairs:
                    out.append((m, {"svm_C": c, "svm_gamma": g}))
            else:
                out.append((m, {}))
        else:
            out.append((m, {}))
    return out
