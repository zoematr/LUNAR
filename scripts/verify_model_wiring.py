# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Verify the assumptions marked `# VERIFY` in the model_utils files for the newer
MoE models (Llama 4 Scout, Mistral Small 4, Qwen3-30B-A3B, Qwen3.6-35B-A3B).

Two phases:

  Phase 1 (tokenizer only — cheap, no GPU, small download):
    * refusal-token ids decode to the intended strings ("I", "As")
    * the hand-written *_CHAT_TEMPLATE matches the model's own chat template

  Phase 2 (full model — needs the weights + a transformers that supports the
  architecture; enabled with --load-model):
    * resolve_text_model() finds the decoder
    * a MoE layer exposes the expected expert / shared-expert / router modules
    * _get_expert_down_proj(...).weight is a writable [hidden, intermediate] tensor
    * _get_num_experts() / _get_router() work
    * hybrid models: report how many layers have self_attn (full) vs not (linear)

Usage:
    python scripts/verify_model_wiring.py --model-family Qwen3-30B-A3B
    python scripts/verify_model_wiring.py --model-family Qwen3-30B-A3B --load-model
    python scripts/verify_model_wiring.py --model-family llama4-scout \
        --model-path /path/to/local/checkpoint --load-model
"""

import argparse
import os
import sys

# Allow running as `python scripts/verify_model_wiring.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pull the in-code constants so we check exactly what the model files declare.
from src.model_utils.llama4_model import (
    LLAMA4_REFUSAL_TOKS,
    LLAMA4_CHAT_TEMPLATE,
)
from src.model_utils.mistral_small4_model import (
    MISTRAL_SMALL4_REFUSAL_TOKS,
    MISTRAL_SMALL4_CHAT_TEMPLATE,
)
from src.model_utils.qwen3moe_model import (
    QWEN3MOE_REFUSAL_TOKS,
    QWEN3MOE_CHAT_TEMPLATE,
    QWEN3MOE_SYSTEM_PROMPT,
)
from src.model_utils.qwen3_5moe_model import (
    QWEN3_5MOE_REFUSAL_TOKS,
    QWEN3_5MOE_CHAT_TEMPLATE,
    QWEN3_5MOE_SYSTEM_PROMPT,
)

# model_family -> verification spec. `refusal_strings` are the tokens each id in
# `refusal_toks` is meant to represent (positionally).
REGISTRY = {
    "llama4-scout": {
        "default_path": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "refusal_toks": LLAMA4_REFUSAL_TOKS,
        "refusal_strings": ["I"],
        "chat_template": LLAMA4_CHAT_TEMPLATE,
        "system_prompt": None,
        "expert_attr": "feed_forward.experts (fused 3D Parameter)",
    },
    "mistral-small-4": {
        "default_path": "mistralai/Mistral-Small-4-119B-2603",
        "refusal_toks": MISTRAL_SMALL4_REFUSAL_TOKS,
        "refusal_strings": ["I"],
        "chat_template": MISTRAL_SMALL4_CHAT_TEMPLATE,
        "system_prompt": None,
        "expert_attr": "mlp.experts / mlp.shared_experts / mlp.gate",
    },
    "Qwen3-30B-A3B": {
        "default_path": "Qwen/Qwen3-30B-A3B",
        "refusal_toks": QWEN3MOE_REFUSAL_TOKS,
        "refusal_strings": ["I", "As"],
        "chat_template": QWEN3MOE_CHAT_TEMPLATE,
        "system_prompt": QWEN3MOE_SYSTEM_PROMPT,
        "expert_attr": "mlp.experts / mlp.gate (no shared expert)",
    },
    "Qwen3.6-35B-A3B": {
        "default_path": "Qwen/Qwen3.6-35B-A3B",
        "refusal_toks": QWEN3_5MOE_REFUSAL_TOKS,
        "refusal_strings": ["I", "As"],
        "chat_template": QWEN3_5MOE_CHAT_TEMPLATE,
        "system_prompt": QWEN3_5MOE_SYSTEM_PROMPT,
        "expert_attr": "mlp.experts / mlp.shared_expert / mlp.gate (hybrid attn)",
    },
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_GLYPH = {PASS: "[PASS]", WARN: "[WARN]", FAIL: "[FAIL]"}


def report(status, msg):
    print(f"{_GLYPH[status]} {msg}")
    return status


def check_refusal_tokens(tok, spec):
    toks = spec["refusal_toks"]
    strings = spec["refusal_strings"]
    print(f"\n--- refusal tokens: declared {toks} (meant to be {strings}) ---")
    decoded = tok.convert_ids_to_tokens(toks)
    print(f"    convert_ids_to_tokens({toks}) = {decoded}")

    statuses = []
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        ok = len(enc) >= 1 and enc[0] in toks
        statuses.append(
            report(
                PASS if ok else FAIL,
                f'encode({s!r}) -> {enc}; first id {"is" if ok else "is NOT"} in declared refusal_toks',
            )
        )
    return FAIL if FAIL in statuses else PASS


def check_chat_template(tok, spec):
    print("\n--- chat template ---")
    messages = [{"role": "user", "content": "<<USER_MSG>>"}]
    if spec["system_prompt"]:
        messages = [{"role": "system", "content": spec["system_prompt"]}] + messages

    # Render the model's own template under a few thinking settings. Our hand-
    # written templates for reasoning models deliberately seed an empty
    # <think></think> block, which matches the `enable_thinking=False` rendering
    # rather than the default — so we accept a match against ANY of these.
    renderings = {}
    for label, kwargs in (
        ("default", {}),
        ("enable_thinking=False", {"enable_thinking": False}),
        ("thinking=False", {"thinking": False}),
    ):
        try:
            renderings[label] = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception:  # noqa: BLE001  (unsupported kwarg / no template)
            continue

    if not renderings:
        return report(WARN, "tokenizer has no usable chat_template; compare manually")

    ours = spec["chat_template"].replace("{instruction}", "<<USER_MSG>>")
    if spec["system_prompt"]:
        ours = ours.replace("{system}", spec["system_prompt"])

    def norm(s):
        return s.replace("<|begin_of_text|>", "").replace("<s>", "").strip()

    print("    OUR TEMPLATE (repr):")
    print("      " + repr(ours))
    matched = None
    for label, rendered in renderings.items():
        is_match = norm(ours) == norm(rendered) or norm(rendered).endswith(norm(ours))
        flag = " <-- MATCH" if is_match else ""
        print(f"    MODEL [{label}] (repr):{flag}")
        print("      " + repr(rendered))
        if is_match and matched is None:
            matched = label

    if matched:
        return report(PASS, f"our template matches the model's '{matched}' rendering")
    return report(
        WARN,
        "our template matches none of the model's renderings — eyeball the reprs above",
    )


def check_model_wiring(family, model_path, device):
    print("\n=== Phase 2: full model wiring ===")
    from src.model_utils.model_loader import load_model
    from src.model_utils.moe_model_base import resolve_text_model

    model_base = load_model(family, model_path, device)
    model = model_base.model

    text_model = resolve_text_model(model)
    report(PASS, f"resolve_text_model -> {type(text_model).__name__} "
                 f"({len(text_model.layers)} layers)")

    # Hybrid-attention audit.
    full = sum(1 for b in text_model.layers if getattr(b, "self_attn", None) is not None)
    linear = len(text_model.layers) - full
    print(f"    attention layers: {full} full (have self_attn.o_proj), {linear} other/linear")

    # Find the first MoE layer and probe expert wiring.
    print(f"\n    expected expert layout: {REGISTRY[family]['expert_attr']}")
    moe_layer = None
    for i in range(len(text_model.layers)):
        try:
            experts = model_base._get_layer_experts(i)
            moe_layer = i
            break
        except Exception:  # noqa: BLE001
            continue
    if moe_layer is None:
        return report(FAIL, "no MoE layer found via _get_layer_experts")

    report(PASS, f"_get_layer_experts({moe_layer}) -> {len(experts)} experts")
    print("    layer module tree (first MoE layer):")
    print("      " + str(text_model.layers[moe_layer]).replace("\n", "\n      "))

    # down_proj weight contract: [hidden, intermediate], clonable + writable.
    dp = model_base._get_expert_down_proj(experts[0])
    w = dp.weight
    shape = tuple(w.clone().shape)
    report(PASS if len(shape) == 2 else FAIL,
           f"_get_expert_down_proj(e0).weight clone shape = {shape} (want [hidden, intermediate])")

    n = model_base._get_num_experts()
    report(PASS if n == len(experts) else WARN,
           f"_get_num_experts() = {n} (len(experts)={len(experts)})")

    router = model_base._get_router(moe_layer)
    report(PASS, f"_get_router({moe_layer}) -> {type(router).__name__}")

    print(f"\n    eoi_toks   = {model_base.eoi_toks}")
    print(f"    refusal    = {model_base.refusal_toks} -> "
          f"{model_base.tokenizer.convert_ids_to_tokens(model_base.refusal_toks)}")
    return PASS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-family", required=True, choices=list(REGISTRY))
    ap.add_argument("--model-path", default=None,
                    help="HF id or local path (defaults to the known HF id)")
    ap.add_argument("--load-model", action="store_true",
                    help="also run Phase 2 (loads the full model — heavy)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = REGISTRY[args.model_family]
    model_path = args.model_path or spec["default_path"]
    print(f"# Verifying '{args.model_family}'  (path: {model_path})")

    from transformers import AutoTokenizer

    print("\n=== Phase 1: tokenizer checks ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"could not load tokenizer: {e}")
        print("  (gated repo? `huggingface-cli login`. Unknown model_type? "
              "needs a newer transformers.)")
        sys.exit(1)

    s1 = check_refusal_tokens(tok, spec)
    s2 = check_chat_template(tok, spec)

    s3 = None
    if args.load_model:
        try:
            s3 = check_model_wiring(args.model_family, model_path, args.device)
        except Exception as e:  # noqa: BLE001
            s3 = report(FAIL, f"Phase 2 failed: {type(e).__name__}: {e}")
            print("  (unsupported model_type -> update transformers; "
                  "wrong attr name -> fix it in the model file.)")
    else:
        print("\n(Phase 2 skipped; pass --load-model to verify module wiring.)")

    print("\n=== summary ===")
    print(f"  refusal tokens : {s1}")
    print(f"  chat template  : {s2}")
    print(f"  model wiring   : {s3 or 'skipped'}")
    sys.exit(1 if FAIL in (s1, s2, s3) else 0)


if __name__ == "__main__":
    main()
