from typing import List, Optional

from validmind import tags, tasks, test

from ._utils import get_client, require


@test("trustyai.SchemaIntegrity")
@tags("trustyai", "data_quality", "documentation", "openshift_ai")
@tasks("classification", "regression")
def SchemaIntegrity(
    model_id: Optional[str] = None,
    expected_inputs: Optional[List[str]] = None,
    expected_outputs: Optional[List[str]] = None,
    require_name_mapping: bool = False,
    trustyai_url: Optional[str] = None,
    trustyai_token: Optional[str] = None,
):
    """Cross-checks the deployed model's observed input/output schema in TrustyAI
    against the features documented in ValidMind.

    ### Purpose

    Model documentation describes the features a model was validated on; the
    serving platform sees the tensors it actually receives. When the two diverge
    — a feature dropped in a new model version, columns reordered, a raw tensor
    name never mapped to a business name — monitoring keeps running but measures
    the wrong thing. This test pins the documented schema to the observed one,
    so a silent contract change between validation and serving fails loudly.

    ### Test Mechanism

    Reads the model's input/output schema (with any applied name mappings) from
    TrustyAI's `GET /info` and compares it against `expected_inputs` /
    `expected_outputs`. Passes when every expected name is observed and, if
    `require_name_mapping` is set, when a human-readable name mapping has been
    applied via `/info/names`.

    ### Signs of High Risk

    - Expected feature names missing from the observed schema
    - Unexpected extra columns — the model is being fed inputs validation never saw
    - Raw tensor names still unmapped in a governed production model
    """
    client = get_client(trustyai_url, trustyai_token)
    entry = client.model_info(
        require(model_id, "model_id", "the inference model's name in TrustyAI (see GET /info)")
    )
    data = entry.get("data", entry)

    def observed_names(schema_key):
        schema = data.get(schema_key) or {}
        items = schema.get("items") or {}
        mapping = schema.get("nameMapping") or {}
        # A mapped column is known by its mapped (business) name; unmapped
        # columns keep their raw tensor name.
        return {mapping.get(raw, raw) for raw in items}, bool(mapping)

    observed_in, mapped_in = observed_names("inputSchema")
    observed_out, mapped_out = observed_names("outputSchema")

    rows = []
    passed = True
    for kind, expected, observed in (
        ("input", expected_inputs or [], observed_in),
        ("output", expected_outputs or [], observed_out),
    ):
        for name in expected:
            found = name in observed
            passed = passed and found
            rows.append({"Kind": kind, "Column": name, "Status": "observed" if found else "MISSING"})
        for name in sorted(observed - set(expected)):
            rows.append({"Kind": kind, "Column": name, "Status": "unexpected (not documented)"})

    if not (expected_inputs or expected_outputs):
        raise ValueError(
            "Provide 'expected_inputs' and/or 'expected_outputs' — the documented "
            "feature and output names to verify against the observed schema."
        )

    if require_name_mapping and not (mapped_in and mapped_out):
        passed = False
        rows.append(
            {
                "Kind": "mapping",
                "Column": "(name mapping)",
                "Status": "missing — apply business names via POST /info/names",
            }
        )

    return {f"Schema of '{model_id}': documented vs observed (TrustyAI)": rows}, passed
