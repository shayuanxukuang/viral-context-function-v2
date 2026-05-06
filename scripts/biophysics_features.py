from __future__ import annotations

import math


BIOPHYSICS_FIELD_NAMES = [
    "bio_tm_helix_count",
    "bio_tm_longest_hydrophobic_run",
    "bio_signal_peptide_score",
    "bio_coiled_coil_score",
    "bio_disorder_score",
    "bio_low_complexity_fraction",
]

HYDROPHOBIC = set("AILMFWVYC")
CHARGED = set("KRDE")
POSITIVE = set("KRH")
DISORDER_PROMOTING = set("RNDQEKGPSAT")
COILED_COIL_ENRICHED = set("LEAIQKV")


def safe_log1p(value: int | float) -> float:
    return math.log1p(max(float(value), 0.0))


def sliding_windows(sequence: str, size: int):
    if size <= 0 or len(sequence) < size:
        return
    for idx in range(len(sequence) - size + 1):
        yield idx, sequence[idx : idx + size]


def fraction_of_set(sequence: str, alphabet: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(1 for aa in sequence if aa in alphabet) / len(sequence)


def longest_hydrophobic_run(sequence: str) -> int:
    longest = 0
    current = 0
    for aa in sequence:
        if aa in HYDROPHOBIC:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def count_tm_windows(sequence: str, window: int = 19, threshold: float = 0.68) -> int:
    count = 0
    last_end = -1
    for idx, chunk in sliding_windows(sequence, window):
        if fraction_of_set(chunk, HYDROPHOBIC) >= threshold and idx >= last_end:
            count += 1
            last_end = idx + window
    return count


def signal_peptide_score(sequence: str) -> float:
    n_term = sequence[:30]
    if len(n_term) < 12:
        return 0.0
    positive_bonus = 1.0 if any(aa in POSITIVE for aa in n_term[:5]) else 0.0
    hydro_scores = [fraction_of_set(chunk, HYDROPHOBIC) for _, chunk in sliding_windows(n_term[:25], 8)]
    hydro_peak = max(hydro_scores) if hydro_scores else 0.0
    cleavage_penalty = 0.0
    if len(n_term) >= 23:
        tail = n_term[15:23]
        small_residues = sum(1 for aa in tail if aa in {"A", "S", "T", "G", "V"})
        cleavage_penalty = min(1.0, small_residues / max(1, len(tail)))
    return max(0.0, min(1.0, 0.55 * hydro_peak + 0.25 * positive_bonus + 0.20 * cleavage_penalty))


def coiled_coil_score(sequence: str) -> float:
    if len(sequence) < 21:
        return 0.0
    best = 0.0
    for _, chunk in sliding_windows(sequence, 21):
        enriched = fraction_of_set(chunk, COILED_COIL_ENRICHED)
        proline_penalty = fraction_of_set(chunk, {"P"})
        charge_balance = fraction_of_set(chunk, CHARGED)
        score = max(0.0, enriched * 0.7 + charge_balance * 0.3 - proline_penalty * 0.8)
        best = max(best, score)
    return min(best, 1.0)


def disorder_score(sequence: str) -> float:
    if not sequence:
        return 0.0
    disorder_fraction = fraction_of_set(sequence, DISORDER_PROMOTING)
    hydrophobic_fraction = fraction_of_set(sequence, HYDROPHOBIC)
    proline_bonus = fraction_of_set(sequence, {"P", "G"})
    score = disorder_fraction * 0.6 + proline_bonus * 0.25 + (1.0 - hydrophobic_fraction) * 0.15
    return max(0.0, min(score, 1.0))


def low_complexity_fraction(sequence: str, window: int = 12) -> float:
    if len(sequence) < window:
        unique = len(set(sequence))
        return 1.0 if unique <= max(1, len(sequence) // 3) else 0.0
    flagged = 0
    total = 0
    for _, chunk in sliding_windows(sequence, window):
        total += 1
        counts = {aa: chunk.count(aa) for aa in set(chunk)}
        entropy = 0.0
        for count in counts.values():
            probability = count / len(chunk)
            entropy -= probability * math.log2(probability)
        if len(counts) <= 4 or entropy <= 1.8:
            flagged += 1
    return flagged / max(1, total)


def compute_biophysics(sequence: str) -> dict[str, float]:
    sequence = sequence.strip().upper()
    tm_run = longest_hydrophobic_run(sequence)
    return {
        "bio_tm_helix_count": float(count_tm_windows(sequence)),
        "bio_tm_longest_hydrophobic_run": safe_log1p(tm_run),
        "bio_signal_peptide_score": signal_peptide_score(sequence),
        "bio_coiled_coil_score": coiled_coil_score(sequence),
        "bio_disorder_score": disorder_score(sequence),
        "bio_low_complexity_fraction": low_complexity_fraction(sequence),
    }
