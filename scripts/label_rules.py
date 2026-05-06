from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LabelRule:
    name: str
    patterns: tuple[str, ...]
    description: str


LABEL_RULES = [
    LabelRule(
        name="polymerase",
        patterns=(
            r"rna-dependent rna polymerase",
            r"dna polymerase",
            r"\bpolymerase\b",
            r"\breplicase\b",
            r"reverse transcriptase",
            r"\btranscriptase\b",
        ),
        description="Polymerase and replicase machinery",
    ),
    LabelRule(name="helicase", patterns=(r"\bhelicase\b",), description="Helicases and unwinding proteins"),
    LabelRule(name="protease", patterns=(r"\bprotease\b", r"\bproteinase\b"), description="Proteases and processing enzymes"),
    LabelRule(
        name="capsid_head",
        patterns=(r"\bcapsid\b", r"\bcoat protein\b", r"\bhead protein\b", r"\bmajor head protein\b"),
        description="Capsid and head structural proteins",
    ),
    LabelRule(
        name="tail_fiber_receptor",
        patterns=(r"tail fiber", r"tail spike", r"baseplate", r"receptor binding"),
        description="Tail fiber, tail spike, and receptor-binding proteins",
    ),
    LabelRule(
        name="tail_assembly",
        patterns=(r"\btail\b", r"tape measure", r"tail assembly"),
        description="Tail assembly and morphogenesis proteins",
    ),
    LabelRule(
        name="portal_terminase_packaging",
        patterns=(r"portal protein", r"terminase", r"packaging"),
        description="Portal, terminase, and genome packaging proteins",
    ),
    LabelRule(
        name="lysis",
        patterns=(r"endolysin", r"\blysin\b", r"\bholin\b", r"\bspanin\b", r"\blysozyme\b"),
        description="Lysis and cell wall disruption proteins",
    ),
    LabelRule(
        name="envelope_glycoprotein",
        patterns=(r"glycoprotein", r"envelope protein", r"spike protein"),
        description="Envelope and glycoprotein structural proteins",
    ),
    LabelRule(
        name="membrane_matrix",
        patterns=(r"membrane protein", r"matrix protein", r"\bmembrane\b"),
        description="Membrane and matrix-associated proteins",
    ),
    LabelRule(name="nucleocapsid", patterns=(r"nucleocapsid",), description="Nucleocapsid proteins"),
    LabelRule(
        name="integrase_recombinase",
        patterns=(r"\bintegrase\b", r"\brecombinase\b"),
        description="Integrases and recombinases",
    ),
    LabelRule(
        name="nuclease",
        patterns=(r"\bnuclease\b", r"endonuclease", r"exonuclease"),
        description="Nucleases and nucleic acid processing proteins",
    ),
    LabelRule(name="methyltransferase", patterns=(r"methyltransferase",), description="Methyltransferases"),
    LabelRule(name="ligase", patterns=(r"\bligase\b",), description="Ligases"),
    LabelRule(
        name="transcription_regulator",
        patterns=(r"transcriptional regulator", r"transcription regulator", r"transactivator"),
        description="Transcriptional regulators and activators",
    ),
    LabelRule(name="polyprotein", patterns=(r"\bpolyprotein\b",), description="Polyprotein precursors"),
]


UNKNOWN_TEXT_MARKERS = ("hypothetical protein", "uncharacterized", "unknown protein")


def normalize_text(row: dict[str, str]) -> str:
    product = row.get("cds_product", "").strip()
    description = row.get("protein_description", "").strip()
    return " ".join(part for part in [product, description] if part).lower()


def label_hits(text: str) -> list[int]:
    hits: list[int] = []
    for idx, rule in enumerate(LABEL_RULES):
        if any(re.search(pattern, text) for pattern in rule.patterns):
            hits.append(idx)
    return hits
