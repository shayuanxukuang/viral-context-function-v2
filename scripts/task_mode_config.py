from __future__ import annotations

from dataclasses import dataclass

from biophysics_features import BIOPHYSICS_FIELD_NAMES
from label_rules import LABEL_RULES


TASK_MODE_ORDER = (
    "protein_only",
    "genome_aware_denovo",
    "annotation_refinement",
)

CONTEXT_BLOCK_ORDER = (
    "local_neighborhood",
    "genome_organization",
    "host_metadata",
    "annotation_context",
    "train_priors",
)

LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS = [
    "context_prev_length_bin",
    "context_next_length_bin",
]

LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS = [
    "context_has_prev_neighbor",
    "context_has_next_neighbor",
    "context_prev_gap_nt",
    "context_next_gap_nt",
    "context_prev_overlap_nt",
    "context_next_overlap_nt",
    "context_same_strand_prev",
    "context_same_strand_next",
]

GENOME_ORGANIZATION_CATEGORY_FIELDS = [
    "context_segment_bucket",
]

GENOME_ORGANIZATION_NUMERIC_FIELDS = [
    "context_log_genome_protein_count",
    "context_relative_order_fraction",
    "context_segment_count",
    "context_log_genome_span_nt",
    "context_orf_density_per_kb",
    "context_genome_mean_protein_length",
    "context_genome_protein_length_cv",
]

HOST_METADATA_CATEGORY_FIELDS = [
    "context_host_supergroup",
]

HOST_METADATA_NUMERIC_FIELDS = [
    "context_log_host_taxid_count",
    "context_log_host_lineage_count",
]

NON_TEXT_CONTEXT_CATEGORY_FIELDS = (
    list(LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS)
    + list(GENOME_ORGANIZATION_CATEGORY_FIELDS)
    + list(HOST_METADATA_CATEGORY_FIELDS)
)

NON_TEXT_CONTEXT_NUMERIC_FIELDS = (
    list(LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS)
    + list(GENOME_ORGANIZATION_NUMERIC_FIELDS)
    + list(HOST_METADATA_NUMERIC_FIELDS)
)

ANNOTATION_CONTEXT_CATEGORY_FIELDS = [
    "context_prev_feature_type",
    "context_next_feature_type",
]

ANNOTATION_CONTEXT_NUMERIC_FIELDS = [
    "context_genome_hypothetical_fraction",
    "context_genome_mat_peptide_fraction",
    "context_prev_is_hypothetical",
    "context_next_is_hypothetical",
]

BASE_REFINEMENT_CATEGORY_FIELDS = [
    "protein_feature_type",
    "source_mol_type",
    "division",
    "host_join_strategy",
]

BASE_REFINEMENT_NUMERIC_FIELDS = [
    "log_host_record_count",
    "log_uniprot_entries",
    "log_uniprot_go_entries",
    "log_uniprot_interpro_entries",
    "log_uniprot_ec_entries",
    "is_hypothetical",
    "is_mat_peptide",
]

CONTEXT_BLOCK_FIELDS = {
    "local_neighborhood": {
        "category_fields": list(LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS),
        "numeric_fields": list(LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS),
    },
    "genome_organization": {
        "category_fields": list(GENOME_ORGANIZATION_CATEGORY_FIELDS),
        "numeric_fields": list(GENOME_ORGANIZATION_NUMERIC_FIELDS),
    },
    "host_metadata": {
        "category_fields": list(HOST_METADATA_CATEGORY_FIELDS),
        "numeric_fields": list(HOST_METADATA_NUMERIC_FIELDS),
    },
    "annotation_context": {
        "category_fields": list(ANNOTATION_CONTEXT_CATEGORY_FIELDS),
        "numeric_fields": list(ANNOTATION_CONTEXT_NUMERIC_FIELDS),
    },
    "train_priors": {
        "category_fields": [],
        "numeric_fields": [],
    },
}


@dataclass(frozen=True)
class FeatureAuditSpec:
    name: str
    source_table: str
    provenance_group: str
    minimum_task_mode: str
    notes: str
    is_model_input_candidate: bool = True
    is_text_derived: bool = False
    is_train_only_stat: bool = False


def allowed_modes(minimum_task_mode: str) -> tuple[str, ...]:
    start = TASK_MODE_ORDER.index(minimum_task_mode)
    return TASK_MODE_ORDER[start:]


def prior_context_numeric_fields() -> list[str]:
    fields: list[str] = []
    for rule in LABEL_RULES:
        fields.append(f"context_train_genome_{rule.name}_fraction")
    for rule in LABEL_RULES:
        fields.append(f"context_train_local_{rule.name}_count")
    return fields


def available_context_blocks(task_mode: str) -> tuple[str, ...]:
    if task_mode == "protein_only":
        return ()
    if task_mode == "genome_aware_denovo":
        return ("local_neighborhood", "genome_organization", "host_metadata")
    if task_mode == "annotation_refinement":
        return (
            "local_neighborhood",
            "genome_organization",
            "host_metadata",
            "annotation_context",
            "train_priors",
        )
    raise ValueError(f"Unsupported task mode: {task_mode}")


def resolve_context_blocks(task_mode: str, requested_blocks: str | list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    allowed = available_context_blocks(task_mode)
    if not requested_blocks:
        return allowed

    if isinstance(requested_blocks, str):
        lowered = requested_blocks.strip().lower()
        if not lowered or lowered in {"default", "all"}:
            return allowed
        tokens = [token.strip() for token in requested_blocks.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in requested_blocks if str(token).strip()]

    unknown = [token for token in tokens if token not in CONTEXT_BLOCK_ORDER]
    if unknown:
        raise ValueError(f"Unknown context block(s): {', '.join(unknown)}")
    disallowed = [token for token in tokens if token not in allowed]
    if disallowed:
        raise ValueError(
            f"Task mode '{task_mode}' does not allow context block(s): {', '.join(disallowed)}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for block_name in CONTEXT_BLOCK_ORDER:
        if block_name in tokens and block_name not in seen:
            ordered.append(block_name)
            seen.add(block_name)
    return tuple(ordered)


def task_mode_feature_lists(
    task_mode: str,
    with_biophysics: bool,
    context_blocks: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    if task_mode not in TASK_MODE_ORDER:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    config = {
        "base_category_fields": [],
        "base_numeric_fields": [],
        "context_category_fields": [],
        "context_numeric_fields": [],
        "biophysics_fields": [],
    }

    selected_blocks = resolve_context_blocks(task_mode, context_blocks)

    if task_mode == "annotation_refinement":
        config["base_category_fields"] = list(BASE_REFINEMENT_CATEGORY_FIELDS)
        config["base_numeric_fields"] = list(BASE_REFINEMENT_NUMERIC_FIELDS)

    for block_name in selected_blocks:
        if block_name == "train_priors":
            config["context_numeric_fields"].extend(prior_context_numeric_fields())
            continue
        block_fields = CONTEXT_BLOCK_FIELDS[block_name]
        config["context_category_fields"].extend(block_fields["category_fields"])
        config["context_numeric_fields"].extend(block_fields["numeric_fields"])

    if with_biophysics:
        config["biophysics_fields"] = list(BIOPHYSICS_FIELD_NAMES)

    return config


def feature_audit_specs() -> list[FeatureAuditSpec]:
    specs = [
        FeatureAuditSpec("protein_sequence", "training_index", "sequence_raw", "protein_only", "Primary amino-acid sequence"),
        FeatureAuditSpec("protein_length_aa", "training_index", "sequence_derived", "protein_only", "Length derived from sequence"),
        FeatureAuditSpec("protein_sequence_sha256", "training_index", "identifier", "protein_only", "Sequence checksum", is_model_input_candidate=False),
        FeatureAuditSpec("protein_accession", "training_index", "identifier", "protein_only", "Record identifier", is_model_input_candidate=False),
        FeatureAuditSpec("genome_accession", "training_index", "identifier", "genome_aware_denovo", "Genome identifier", is_model_input_candidate=False),
        FeatureAuditSpec("genome_version", "training_index", "identifier", "genome_aware_denovo", "Genome version", is_model_input_candidate=False),
        FeatureAuditSpec("virus_tax_id", "training_index", "taxonomy_identifier", "annotation_refinement", "Virus taxonomic identifier", is_model_input_candidate=False),
        FeatureAuditSpec("virus_name", "training_index", "taxonomy_text", "annotation_refinement", "Virus name text", is_model_input_candidate=False),
        FeatureAuditSpec("virus_lineage", "training_index", "taxonomy_text", "annotation_refinement", "Virus lineage text", is_model_input_candidate=False),
        FeatureAuditSpec("protein_description", "training_index", "annotation_text", "annotation_refinement", "Protein free-text description", is_text_derived=True, is_model_input_candidate=False),
        FeatureAuditSpec("cds_product", "training_index", "annotation_text", "annotation_refinement", "CDS product text", is_text_derived=True, is_model_input_candidate=False),
        FeatureAuditSpec("cds_gene", "training_index", "annotation_text", "annotation_refinement", "Gene symbol text", is_text_derived=True, is_model_input_candidate=False),
        FeatureAuditSpec("cds_locus_tag", "training_index", "identifier", "annotation_refinement", "Local locus tag", is_model_input_candidate=False),
        FeatureAuditSpec("protein_organism", "training_index", "annotation_text", "annotation_refinement", "Protein organism text", is_text_derived=True, is_model_input_candidate=False),
        FeatureAuditSpec("source_mol_type", "training_index", "annotation_metadata", "annotation_refinement", "Molecule type from record"),
        FeatureAuditSpec("source_segment", "training_index", "genome_structure", "genome_aware_denovo", "Segment identifier"),
        FeatureAuditSpec("source_isolate", "training_index", "sample_metadata_text", "annotation_refinement", "Isolate text", is_model_input_candidate=False),
        FeatureAuditSpec("source_host", "training_index", "sample_metadata_text", "annotation_refinement", "Source host text", is_model_input_candidate=False),
        FeatureAuditSpec("source_geo_loc_name", "training_index", "sample_metadata_text", "annotation_refinement", "Geolocation text", is_model_input_candidate=False),
        FeatureAuditSpec("protein_feature_type", "training_index", "annotation_feature_type", "annotation_refinement", "Feature kind such as CDS/mat_peptide"),
        FeatureAuditSpec("cds_location_raw", "training_index", "genome_coordinates", "genome_aware_denovo", "Raw genomic location string", is_model_input_candidate=False),
        FeatureAuditSpec("cds_location_kind", "training_index", "genome_coordinates", "genome_aware_denovo", "Joined/complement location kind", is_model_input_candidate=False),
        FeatureAuditSpec("cds_start", "training_index", "genome_coordinates", "genome_aware_denovo", "CDS start coordinate"),
        FeatureAuditSpec("cds_end", "training_index", "genome_coordinates", "genome_aware_denovo", "CDS end coordinate"),
        FeatureAuditSpec("cds_strand", "training_index", "genome_coordinates", "genome_aware_denovo", "CDS strand"),
        FeatureAuditSpec("cds_part_count", "training_index", "genome_coordinates", "genome_aware_denovo", "Part count for joined CDS"),
        FeatureAuditSpec("cds_partial_left", "training_index", "genome_coordinates", "genome_aware_denovo", "Left truncation flag"),
        FeatureAuditSpec("cds_partial_right", "training_index", "genome_coordinates", "genome_aware_denovo", "Right truncation flag"),
        FeatureAuditSpec("host_join_strategy", "training_index", "annotation_pipeline", "annotation_refinement", "How host metadata was joined"),
        FeatureAuditSpec("host_record_count", "training_index", "host_taxonomy", "genome_aware_denovo", "Host association count"),
        FeatureAuditSpec("host_tax_ids_json", "training_index", "host_taxonomy", "genome_aware_denovo", "Host taxon identifiers", is_model_input_candidate=False),
        FeatureAuditSpec("host_names_json", "training_index", "host_taxonomy_text", "annotation_refinement", "Host names", is_model_input_candidate=False),
        FeatureAuditSpec("host_lineages_json", "training_index", "host_taxonomy", "genome_aware_denovo", "Host lineage values", is_model_input_candidate=False),
        FeatureAuditSpec("host_evidence_json", "training_index", "host_taxonomy", "annotation_refinement", "Host evidence terms", is_model_input_candidate=False),
        FeatureAuditSpec("host_pmids_json", "training_index", "host_taxonomy", "annotation_refinement", "Host evidence PMIDs", is_model_input_candidate=False),
        FeatureAuditSpec("host_sample_types_json", "training_index", "sample_metadata_text", "annotation_refinement", "Host sample types", is_model_input_candidate=False),
        FeatureAuditSpec("host_source_organisms_json", "training_index", "sample_metadata_text", "annotation_refinement", "Host source organisms", is_model_input_candidate=False),
        FeatureAuditSpec("reviewed_uniprot_entries_for_taxon", "training_index", "knowledgebase_summary", "annotation_refinement", "UniProt reviewed count"),
        FeatureAuditSpec("reviewed_uniprot_entries_with_go_for_taxon", "training_index", "knowledgebase_summary", "annotation_refinement", "UniProt GO count"),
        FeatureAuditSpec("reviewed_uniprot_entries_with_interpro_for_taxon", "training_index", "knowledgebase_summary", "annotation_refinement", "UniProt InterPro count"),
        FeatureAuditSpec("reviewed_uniprot_entries_with_ec_for_taxon", "training_index", "knowledgebase_summary", "annotation_refinement", "UniProt EC count"),
        FeatureAuditSpec("log_protein_length", "derived_numeric", "sequence_derived", "protein_only", "Log protein length"),
        FeatureAuditSpec("log_host_record_count", "derived_numeric", "host_taxonomy", "annotation_refinement", "Log host record count"),
        FeatureAuditSpec("log_uniprot_entries", "derived_numeric", "knowledgebase_summary", "annotation_refinement", "Log UniProt reviewed count"),
        FeatureAuditSpec("log_uniprot_go_entries", "derived_numeric", "knowledgebase_summary", "annotation_refinement", "Log UniProt GO count"),
        FeatureAuditSpec("log_uniprot_interpro_entries", "derived_numeric", "knowledgebase_summary", "annotation_refinement", "Log UniProt InterPro count"),
        FeatureAuditSpec("log_uniprot_ec_entries", "derived_numeric", "knowledgebase_summary", "annotation_refinement", "Log UniProt EC count"),
        FeatureAuditSpec("is_hypothetical", "derived_numeric", "annotation_text", "annotation_refinement", "Derived from hypothetical/uncharacterized text marker", is_text_derived=True),
        FeatureAuditSpec("is_mat_peptide", "derived_numeric", "annotation_feature_type", "annotation_refinement", "Derived from feature type"),
    ]

    for field in BIOPHYSICS_FIELD_NAMES:
        specs.append(
            FeatureAuditSpec(
                field,
                "biophysics",
                "sequence_derived_biophysics",
                "protein_only",
                "Cheap sequence-derived biophysical heuristic",
            )
        )

    for field in LOCAL_NEIGHBORHOOD_CATEGORY_FIELDS:
        specs.append(FeatureAuditSpec(field, "context_splitaware", "local_neighborhood_context", "genome_aware_denovo", "Local neighborhood category"))

    for field in LOCAL_NEIGHBORHOOD_NUMERIC_FIELDS:
        specs.append(FeatureAuditSpec(field, "context_splitaware", "local_neighborhood_context", "genome_aware_denovo", "Local neighborhood numeric"))

    for field in GENOME_ORGANIZATION_CATEGORY_FIELDS:
        notes = "Non-text context category"
        if field == "context_segment_bucket":
            notes = "Genome segment bucket"
        specs.append(FeatureAuditSpec(field, "context_splitaware", "genome_organization_context", "genome_aware_denovo", notes))

    for field in GENOME_ORGANIZATION_NUMERIC_FIELDS:
        specs.append(FeatureAuditSpec(field, "context_splitaware", "genome_organization_context", "genome_aware_denovo", "Genome-scale organization numeric"))

    for field in HOST_METADATA_CATEGORY_FIELDS:
        specs.append(
            FeatureAuditSpec(
                field,
                "context_splitaware",
                "host_metadata_context",
                "genome_aware_denovo",
                "Collapsed host taxonomy supergroup",
            )
        )

    for field in HOST_METADATA_NUMERIC_FIELDS:
        specs.append(
            FeatureAuditSpec(
                field,
                "context_splitaware",
                "host_metadata_context",
                "genome_aware_denovo",
                "Host metadata count feature",
            )
        )

    for field in ANNOTATION_CONTEXT_CATEGORY_FIELDS:
        specs.append(
            FeatureAuditSpec(
                field,
                "context_splitaware",
                "annotation_context",
                "annotation_refinement",
                "Neighbor annotation state",
            )
        )

    for field in ANNOTATION_CONTEXT_NUMERIC_FIELDS:
        specs.append(
            FeatureAuditSpec(
                field,
                "context_splitaware",
                "annotation_context",
                "annotation_refinement",
                "Annotation-derived context numeric",
                is_text_derived=field.endswith("is_hypothetical") or "hypothetical" in field,
            )
        )

    for field in prior_context_numeric_fields():
        specs.append(
            FeatureAuditSpec(
                field,
                "context_splitaware",
                "train_fold_weak_label_prior",
                "annotation_refinement",
                "Weak-label prior built from train partition only",
                is_text_derived=True,
                is_train_only_stat=True,
            )
        )

    return specs
