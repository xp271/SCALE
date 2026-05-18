import torch
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
import argparse
import logging
import os
import sys
import json
from pathlib import Path
import numpy as np

# Ensure repo root (LLM-sycophancy) is on path for utils.standalone_fake_quant
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent  # experiments -> LLM-sycophancy
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set up logging
logging.basicConfig(filename='inference.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_hf_token(cli_token: str | None) -> str | None:
    """HF Hub token: ``--hf_token`` > ``HF_TOKEN`` > ``HUGGING_FACE_HUB_TOKEN``."""
    if cli_token is not None and str(cli_token).strip():
        return str(cli_token).strip()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference on a dataset with pre-constructed questions.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path (e.g. fake_quant_model dir)")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base HF model name when loading from fake_quant dir (required if model_name is fake_quant)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "commonsenseqa"],
        help="Dataset name for output paths (e.g. mmlu, commonsenseqa); input pkl via --input_filename",
    )
    parser.add_argument("--prefix_type", type=str, default="", 
                        choices=["", "academic", "behavior"], 
                        help="Type of prefix used (e.g., 'academic', 'behavior').")
    parser.add_argument("--academic_level", type=str, default="", 
                        choices=["", "beginner", "intermediate", "advanced"], 
                        help="Academic level for academic prefix (beginner, intermediate, advanced). "
                             "Only applies when prefix_type='academic'.")
    parser.add_argument("--prefix_subtype", type=str, default="original", 
                        choices=["", "original", "mixing_subject", "third_pov"], 
                        help="Subtype of prefix (original, mixing_subject, third_pov).")
    parser.add_argument("--question_type", type=str, default="plain", 
                        choices=["prefix_and_opinion", "opinion_only", "plain"], 
                        help="Type of the questions: 'prefix_and_opinion' (prefix + opinion), "
                             "'opinion_only' (just opinion), or 'plain' (no prefix or opinion).")
    parser.add_argument("--input_filename", type=str, default="../../lib/plain/mmlu_plain.pkl",
                        help="Input .pkl file with pre-constructed questions")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Data seed: use dataset with this seed (e.g. mmlu_plain_42.pkl). Output pkl will also include seed in filename.")
    parser.add_argument("--model_output_name", type=str, default=None,
                        help="Basename for output pkl (e.g. model_id_method_id). If set, used instead of deriving from model path.")
    parser.add_argument("--full_question_column", type=str, default="full_question",
                        help="Name of the column containing the full question text in the input DataFrame")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device to run inference on. "
            "Use 'auto' (default) to select CUDA if available, "
            "or specify e.g. 'cuda', 'cuda:1', or 'cpu'."
        ),
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        help="Forward with output_hidden_states=True and store per-layer vectors (uses much more VRAM).",
    )
    parser.add_argument(
        "--answer_max_new_tokens",
        type=int,
        default=8,
        help="Generate up to N new tokens, then extract option letter from generated text. Set 0 to disable and use argmax-only.",
    )
    parser.add_argument(
        "--debug_inference",
        action="store_true",
        help="Enable full inference debug (checkpoint, load, tokenizer, smoke, per-sample truncate/first-step logits/generate; extra logits on invalid). Or env RUN_SYCO_DEBUG=1/true.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Hugging Face Hub token for gated/private models; overrides HF_TOKEN / HUGGING_FACE_HUB_TOKEN env if set. "
            "run_pipeline loads this from config/pipeline_config.yaml syco.huggingface_token into the subprocess environment."
        ),
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="output",
        help=(
            "Output root directory for pkl files (relative to cwd or absolute). "
            "Final path is {output_base}/{dataset}/{question_type}[/{prefix_type}/{prefix_subtype}/{academic_level}]/{basename}[_{seed}].pkl. "
            "Default: 'output' (preserves the historical layout)."
        ),
    )
    return parser.parse_args()


def _env_inference_debug_enabled() -> bool:
    raw = os.environ.get("RUN_SYCO_DEBUG", "").strip()
    if not raw:
        return False
    low = raw.lower()
    if low in ("0", "false", "no", "off"):
        return False
    return True


def inference_debug_enabled(args) -> bool:
    return bool(getattr(args, "debug_inference", False)) or _env_inference_debug_enabled()


def syco_debug(msg: str, dbg: bool) -> None:
    if dbg:
        print(f"[syco_debug] {msg}")


def checkpoint_precheck_fake_quant(model_dir: str, dbg: bool) -> None:
    if not dbg:
        return
    p = Path(model_dir).resolve()
    syco_debug(f"fake_quant_dir (absolute)={p}", dbg)
    syco_debug(f"exists config.json={(p / 'config.json').exists()}", dbg)
    st = sorted(p.glob("*.safetensors"))
    syco_debug(f"*.safetensors count={len(st)} names={[x.name for x in st[:6]]}{'...' if len(st) > 6 else ''}", dbg)
    syco_debug(f"pytorch_model.bin exists={(p / 'pytorch_model.bin').exists()}", dbg)
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        syco_debug("no model.safetensors.index.json (single-shard or bin layout possible)", dbg)
        return
    try:
        with open(idx, encoding="utf-8") as f:
            meta = json.load(f)
        wm = meta.get("weight_map", {})
        keys = list(wm.keys())
        n_buf = sum(1 for k in keys if "buf_" in k)
        syco_debug(f"index.json weight_map total_keys={len(keys)} keys_with_buf_substring={n_buf}", dbg)
        shards = sorted(set(wm.values()))
        syco_debug(f"index shard files ({len(shards)}): {shards[:8]}{'...' if len(shards) > 8 else ''}", dbg)
        missing_shard = [s for s in shards if not (p / s).exists()]
        if missing_shard:
            syco_debug(f"MISSING shard files on disk: {missing_shard}", dbg)
        else:
            syco_debug("all index-listed shard files present on disk", dbg)
    except Exception as e:
        syco_debug(f"failed to read index.json: {e}", dbg)


def _logits_topk_and_entropy(logits_1v: torch.Tensor, tokenizer, k: int = 5):
    logits_1v = logits_1v.detach().float()
    probs = torch.softmax(logits_1v, dim=-1)
    logp = torch.log_softmax(logits_1v, dim=-1)
    ent = float(-(probs * logp).sum().item())
    topv, topi = torch.topk(logits_1v, k)
    rows = []
    for j in range(k):
        tid = int(topi[j].item())
        dec = tokenizer.decode([tid], skip_special_tokens=False)
        rows.append(f"id={tid} logit={float(topv[j]):.4f} decode_no_skip={repr(dec)}")
    return ent, rows


def run_inference_smoke_test(model, tokenizer, device, dbg: bool) -> None:
    if not dbg:
        return
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :].squeeze(0)
        ent, rows = _logits_topk_and_entropy(logits, tokenizer, 5)
        argmax_id = int(torch.argmax(logits).item())
        syco_debug(f"smoke prompt={repr(prompt)}", dbg)
        syco_debug(f"smoke last-position entropy={ent:.6f} top-5:", dbg)
        for r in rows:
            syco_debug(f"  {r}", dbg)
        gen = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_ids = gen[0][inputs["input_ids"].shape[1]:].tolist()
        first_new = int(new_ids[0]) if new_ids else None
        syco_debug(
            f"smoke greedy first_new_token_id={first_new} argmax_prompt_id={argmax_id} "
            f"match={first_new == argmax_id if first_new is not None else 'n/a'}",
            dbg,
        )
        if new_ids:
            raw_dec = tokenizer.decode(new_ids, skip_special_tokens=False)
            syco_debug(f"smoke decode_new_8tok_no_skip={repr(raw_dec)}", dbg)


def tokenizer_special_debug(tokenizer, dbg: bool) -> None:
    if not dbg:
        return
    syco_debug(
        f"tokenizer eos={getattr(tokenizer, 'eos_token_id', None)} "
        f"pad={getattr(tokenizer, 'pad_token_id', None)} "
        f"bos={getattr(tokenizer, 'bos_token_id', None)}",
        dbg,
    )
    for tid in (235248,):  # common Gemma space-like collapse id from logs; harmless if absent
        try:
            a = tokenizer.decode([tid], skip_special_tokens=False)
            b = tokenizer.decode([tid], skip_special_tokens=True)
            syco_debug(f"probe token_id={tid} decode_no_skip={repr(a)} decode_skip_special={repr(b)}", dbg)
        except Exception as e:
            syco_debug(f"probe token_id={tid} failed: {e}", dbg)

# Debug: print raw generation for first few invalid
_DEBUG_INVALID_COUNT = [0]
DEBUG_INVALID_MAX_PRINT = 8

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()


def option_letters_from_question(question: str) -> list[str]:
    """
    Parse contiguous option labels from question body, e.g. A./B./C... .
    Returns a dynamic list like ['A','B','C'] or ['A',...,'E']; fallback to A-D.
    """
    if not isinstance(question, str):
        return ["A", "B", "C", "D"]
    core = question
    for sep in ("\nAnswer:", "\nI believe the answer is"):
        if sep in core:
            core = core.split(sep, 1)[0]
    letters: list[str] = []
    expected = 0
    for line in core.split("\n"):
        m = re.match(r"^\s*([A-Z])\.\s+", line)
        if not m:
            continue
        L = m.group(1)
        if ord(L) - ord("A") == expected:
            letters.append(L)
            expected += 1
    if len(letters) >= 2 and letters == [chr(ord("A") + i) for i in range(len(letters))]:
        return letters
    return ["A", "B", "C", "D"]


def extract_answer_letter(raw_text: str, expected_letters: list[str]) -> str:
    """
    Extract first valid uppercase option letter from generated text.
    Prioritize strict patterns like {A}, then standalone letters.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""
    s = raw_text.strip()
    if not s:
        return ""
    allowed = set(expected_letters)

    m = re.search(r"\{([A-Z])\}", s)
    if m and m.group(1) in allowed:
        return m.group(1)

    # standalone uppercase letter boundary; avoid matching inside words
    for ch in re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", s):
        if ch in allowed:
            return ch
    return ""


def process_question(
    question,
    tokenizer,
    model,
    device,
    save_hidden_states: bool = False,
    answer_max_new_tokens: int = 8,
    debug_inference: bool = False,
    sample_index: int | None = None,
):
    """Process a single question and return the model's answer and optional layer-wise hidden states."""
    try:
        prompt = f"Question: ||{question}||\nRespond with exactly one uppercase letter (A, B, C, D, etc.) and nothing else.\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if debug_inference:
            ids1 = inputs["input_ids"][0]
            full_dec = tokenizer.decode(ids1, skip_special_tokens=False)
            tag = f"sample[{sample_index}]" if sample_index is not None else "sample"
            syco_debug(
                f"{tag} input_ids.shape={tuple(inputs['input_ids'].shape)} "
                f"attention_mask_sum={int(inputs['attention_mask'].sum()) if 'attention_mask' in inputs else 'n/a'}",
                debug_inference,
            )
            syco_debug(
                f"{tag} decoded_len={len(full_dec)} head={full_dec[:200]!r} tail={full_dec[-200:]!r}",
                debug_inference,
            )

        expected_letters = option_letters_from_question(question)
        pred_text = ""
        result = ""
        pred_token_id = None
        gen_token_ids: list[int] = []
        gen_text_raw = ""
        argmax_text_raw = ""

        with torch.no_grad():
            # Prefer a short generation then parse answer letter.
            if answer_max_new_tokens and answer_max_new_tokens > 0:
                gen_kw = dict(
                    max_new_tokens=int(answer_max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if debug_inference:
                    gen_kw["return_dict_in_generate"] = True
                    gen_kw["output_scores"] = True
                gen_out = model.generate(**inputs, **gen_kw)
                gen_seq = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
                gen_tokens = gen_seq[0][inputs["input_ids"].shape[1]:]
                gen_token_ids = [int(x) for x in gen_tokens.tolist()]
                gen_text_raw = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                result = extract_answer_letter(pred_text, expected_letters)
                tag = f"sample[{sample_index}]" if sample_index is not None else "sample"
                if debug_inference and hasattr(gen_out, "scores") and gen_out.scores:
                    s0 = gen_out.scores[0]
                    li = s0.reshape(-1, s0.shape[-1])[0]
                    ent, rows = _logits_topk_and_entropy(li, tokenizer, 5)
                    syco_debug(f"{tag} generate step0 entropy={ent:.6f} top-5:", debug_inference)
                    for r in rows:
                        syco_debug(f"  {r}", debug_inference)

            # Fallback: argmax of next-token logits
            outputs = model(**inputs, output_hidden_states=save_hidden_states)
            if not result:
                logits = outputs.logits[:, -1, :]
                pred_token_id = int(torch.argmax(logits, dim=-1).item())
                argmax_text_raw = tokenizer.decode([pred_token_id], skip_special_tokens=False)
                pred_text = tokenizer.decode([pred_token_id], skip_special_tokens=True).strip()
                if pred_text and pred_text[0].isupper() and pred_text[0].isalpha():
                    cand = pred_text[0]
                    result = cand if cand in expected_letters else ""
                else:
                    cand = pred_text if is_valid_answer(pred_text) else ""
                    result = cand if cand in expected_letters else ""

        if debug_inference and gen_token_ids and answer_max_new_tokens:
            with torch.no_grad():
                li2 = model(**inputs).logits[:, -1, :].squeeze(0)
                am2 = int(torch.argmax(li2).item())
            tag = f"sample[{sample_index}]" if sample_index is not None else "sample"
            syco_debug(
                f"{tag} single_forward_argmax_id={am2} vs first_gen_id={gen_token_ids[0]} "
                f"match={am2 == gen_token_ids[0]}",
                debug_inference,
            )

        if not is_valid_answer(result):
            logging.warning(f"Invalid output for question: '{question[:50]}...'. Pred: '{pred_text}'")
            if _DEBUG_INVALID_COUNT[0] < DEBUG_INVALID_MAX_PRINT:
                _DEBUG_INVALID_COUNT[0] += 1
                if debug_inference:
                    syco_debug(
                        f"invalid #{_DEBUG_INVALID_COUNT[0]} sample_index={sample_index} repr={repr(pred_text)} len={len(pred_text)}",
                        debug_inference,
                    )
                    syco_debug(
                        f"invalid expected={expected_letters} answer_max_new_tokens={answer_max_new_tokens} "
                        f"pred_token_id={pred_token_id}",
                        debug_inference,
                    )
                    if gen_token_ids:
                        syco_debug(
                            f"invalid gen_token_ids(first20)={gen_token_ids[:20]} gen_raw_no_skip={repr(gen_text_raw)}",
                            debug_inference,
                        )
                    if pred_token_id is not None:
                        syco_debug(f"invalid argmax_raw_no_skip={repr(argmax_text_raw)}", debug_inference)
                    raw_strip = gen_text_raw.strip() if gen_text_raw else ""
                    syco_debug(
                        f"invalid strip_effect raw_len={len(gen_text_raw)} stripped_len={len(raw_strip)} "
                        f"pred_after_strip={repr(pred_text)}",
                        debug_inference,
                    )
                    with torch.no_grad():
                        logits_inv = model(**inputs).logits[:, -1, :].squeeze(0)
                        ent_inv, rows_inv = _logits_topk_and_entropy(logits_inv, tokenizer, 5)
                    syco_debug(f"invalid last-pos logits entropy={ent_inv:.6f} top-5:", debug_inference)
                    for r in rows_inv:
                        syco_debug(f"  {r}", debug_inference)
                    if gen_token_ids:
                        am_inv = int(torch.argmax(logits_inv).item())
                        syco_debug(
                            f"invalid argmax_id={am_inv} first_gen_id={gen_token_ids[0]} match={am_inv == gen_token_ids[0]}",
                            debug_inference,
                        )
                else:
                    print(f"[DEBUG] invalid raw output #{_DEBUG_INVALID_COUNT[0]}: repr={repr(pred_text)} len={len(pred_text)}")
                    print(
                        f"[DEBUG] expected={expected_letters} "
                        f"answer_max_new_tokens={answer_max_new_tokens} pred_token_id={pred_token_id}"
                    )
                    if gen_token_ids:
                        print(
                            f"[DEBUG] gen_token_ids(first20)={gen_token_ids[:20]} "
                            f"gen_raw_no_skip={repr(gen_text_raw)}"
                        )
                    if pred_token_id is not None:
                        print(f"[DEBUG] argmax_raw_no_skip={repr(argmax_text_raw)}")

        layer_hidden_states = {}
        if save_hidden_states and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            total_layers = len(hidden_states) - 1
            for layer_idx in range(total_layers):
                hs = hidden_states[layer_idx + 1][:, -1, :]
                try:
                    hs_vec = hs.detach().cpu().to(torch.float16).numpy()
                except Exception:
                    hs_vec = hs.detach().cpu().numpy()
                layer_hidden_states[f"layer_{layer_idx}"] = hs_vec

        return result, layer_hidden_states
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        if _DEBUG_INVALID_COUNT[0] < DEBUG_INVALID_MAX_PRINT:
            _DEBUG_INVALID_COUNT[0] += 1
            import traceback
            if debug_inference:
                syco_debug(
                    f"process_question exception #{_DEBUG_INVALID_COUNT[0]}: {type(e).__name__}: {e}",
                    debug_inference,
                )
                syco_debug(traceback.format_exc(), debug_inference)
            else:
                print(f"[DEBUG] process_question exception #{_DEBUG_INVALID_COUNT[0]}: {type(e).__name__}: {e}")
                print(traceback.format_exc())
        return "Error", {}

def main():
    args = parse_args()
    dbg = inference_debug_enabled(args)
    if dbg:
        syco_debug(
            f"inference debug ON (CLI --debug_inference={getattr(args, 'debug_inference', False)}, "
            f"RUN_SYCO_DEBUG={os.environ.get('RUN_SYCO_DEBUG', '')!r})",
            dbg,
        )

    model_name = args.model_name
    base_model_name = getattr(args, "base_model_name", None)
    dataset = args.dataset
    prefix_type = args.prefix_type
    academic_level = args.academic_level
    prefix_subtype = args.prefix_subtype
    question_type = args.question_type
    input_filename = args.input_filename
    data_seed = getattr(args, "data_seed", None)
    full_question_column = args.full_question_column
    max_retries = args.max_retries
    device_arg = args.device
    save_hidden_states = getattr(args, "save_hidden_states", False)
    answer_max_new_tokens = getattr(args, "answer_max_new_tokens", 8)

    if data_seed is not None:
        base, ext = os.path.splitext(input_filename)
        base = re.sub(r"_\d+$", "", base)
        input_filename = f"{base}_{data_seed}{ext}"
        print(f"Using seed {data_seed} dataset: {input_filename}")

    # Validation: Ensure academic_level is only specified for academic prefix
    if academic_level and prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")

    if question_type == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")

    hf_token = resolve_hf_token(getattr(args, "hf_token", None))
    if not hf_token:
        print("Warning: HF_TOKEN is not set. Will try to load the model without authentication (only works for public models).")
        logging.warning("HF_TOKEN is not set. Attempting anonymous loading from Hugging Face Hub.")

    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # With cuda:N, use device_map at load to avoid OOM from allocating on cuda:0 first
    load_on_specific_device = (
        isinstance(device_arg, str) and device_arg.startswith("cuda:") and device_arg != "cuda"
    )

    if dbg:
        syco_debug(
            f"device_arg={repr(device_arg)}, load_on_specific_device={load_on_specific_device}",
            dbg,
        )
        if torch.cuda.is_available():
            syco_debug(
                f"torch.cuda.current_device()={torch.cuda.current_device()}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}",
                dbg,
            )
    else:
        print(f"[DEBUG] device_arg={repr(device_arg)}, load_on_specific_device={load_on_specific_device}")
        if torch.cuda.is_available():
            print(
                f"[DEBUG] torch.cuda.current_device()={torch.cuda.current_device()}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}"
            )

    try:
        # If cuda:N set, set current device to N first so libs do not alloc on cuda:0
        if load_on_specific_device:
            gpu_id = int(device_arg.split(":")[1])
            torch.cuda.set_device(gpu_id)
            if dbg:
                syco_debug(
                    f"set default GPU to {gpu_id}, current_device now={torch.cuda.current_device()}",
                    dbg,
                )
            else:
                print(f"[DEBUG] set default GPU to {gpu_id}, current_device now={torch.cuda.current_device()}")

        model_path = Path(model_name).resolve()
        use_fake_quant_loader = (
            base_model_name is not None
            and (
                "fake_quant_model" in model_name
                or (
                    model_path.is_dir()
                    and (model_path / "config.json").exists()
                    and (any(model_path.glob("*.safetensors")) or (model_path / "pytorch_model.bin").exists())
                )
            )
        )
        if use_fake_quant_loader:
            from utils.standalone_fake_quant import load_model_for_fake_quant
            print("Loading tokenizer and model (fake_quant path)...")
            checkpoint_precheck_fake_quant(model_name, dbg)
            tokenizer_kwargs = {"trust_remote_code": True}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            model, tokenizer = load_model_for_fake_quant(
                model_name,
                base_model_name,
                device=device if not load_on_specific_device else None,
                device_map=device_arg if load_on_specific_device else None,
                token=hf_token,
                debug_inference=dbg,
            )
            if not load_on_specific_device:
                model = model.to(device)
        else:
            print("Loading tokenizer...")
            tokenizer_kwargs = {"trust_remote_code": True}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
            tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

            print("Loading model...")
            model_kwargs = {"trust_remote_code": True}
            if hf_token:
                model_kwargs["token"] = hf_token
            if load_on_specific_device:
                model_kwargs["device_map"] = device_arg
                syco_debug(f"from_pretrained(..., device_map={repr(device_arg)})", dbg)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if not load_on_specific_device:
                model = model.to(device)
        try:
            model_device = next(model.parameters()).device
            syco_debug(f"model first param device: {model_device}", dbg)
        except StopIteration:
            syco_debug("model has no parameters (unexpected)", dbg)
        model.eval()
        tokenizer_special_debug(tokenizer, dbg)
        run_inference_smoke_test(model, tokenizer, device, dbg)

        # Load the pre-constructed DataFrame
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")
        logging.info(f"Loaded DataFrame with {len(df)} entries from {input_filename}.")

        if full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame '{input_filename}' must contain a '{full_question_column}' column.")

        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "layer_hidden_states" not in df.columns:
            df["layer_hidden_states"] = None

        _DEBUG_INVALID_COUNT[0] = 0

        # Process each question
        questions = df[full_question_column].tolist()
        for i, question in tqdm(enumerate(questions), total=len(questions), desc="Initial processing"):
            if not is_valid_answer(df.at[i, "model_answer"]):
                answer, layer_hidden_states = process_question(
                    question, tokenizer, model, device,
                    save_hidden_states=save_hidden_states,
                    answer_max_new_tokens=answer_max_new_tokens,
                    debug_inference=dbg,
                    sample_index=i,
                )
                df.at[i, "model_answer"] = answer
                df.at[i, "layer_hidden_states"] = layer_hidden_states

        # Retry loop for invalid answers
        retry_count = 0
        while retry_count < max_retries:
            invalid_indices = df.index[
                df["model_answer"].isna() |
                (df["model_answer"] == "") |
                (df["model_answer"] == "Error") |
                (~df["model_answer"].apply(is_valid_answer))
            ].tolist()

            if not invalid_indices:
                print("All entries have valid answers!")
                logging.info("All entries have valid answers!")
                break

            print(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            logging.info(f"Retry {retry_count + 1}/{max_retries}: Found {len(invalid_indices)} entries with invalid answers.")
            # Debug: print index, model_answer, question summary for first invalid samples
            n_show = min(10, len(invalid_indices))
            print(f"[DEBUG] First {n_show} invalid samples (index, model_answer, question_preview):")
            for idx in invalid_indices[:n_show]:
                ans = df.at[idx, "model_answer"]
                q = df.at[idx, full_question_column]
                q_preview = (q[:70] + "…") if isinstance(q, str) and len(q) > 70 else (q or "")
                print(f"  idx={idx} model_answer={repr(ans)} question={repr(q_preview)}")
            for idx in tqdm(invalid_indices, desc=f"Retry {retry_count + 1}"):
                question = df.at[idx, full_question_column]
                answer, layer_hidden_states = process_question(
                    question, tokenizer, model, device,
                    save_hidden_states=save_hidden_states,
                    answer_max_new_tokens=answer_max_new_tokens,
                    debug_inference=dbg,
                    sample_index=idx,
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_hidden_states"] = layer_hidden_states

            retry_count += 1
            time.sleep(1)

        # Construct output directory dynamically: {output_base}/{dataset}/{question_type}/{prefix_type}/{prefix_subtype}/{academic_level}
        _output_base = getattr(args, "output_base", None) or "output"
        output_dir_parts = [f"{_output_base}/{dataset}"]
        if question_type:
            output_dir_parts.append(question_type)
        if prefix_type:
            output_dir_parts.append(prefix_type)
            output_dir_parts.append(prefix_subtype)
            if prefix_type == "academic":
                output_dir_parts.append(academic_level)
        output_dir = os.path.join(*output_dir_parts)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Output filename: use model_output_name if provided, else derive from model path; include data_seed when specified
        _bn = getattr(args, "model_output_name", None) or model_name.split("/")[-1].replace(".", "_")
        output_basename = _bn.replace("\\", "_").replace("/", "_")
        if data_seed is not None:
            output_filename = f"{output_dir}/{output_basename}_{data_seed}.pkl"
        else:
            output_filename = f"{output_dir}/{output_basename}.pkl"

        # Check for invalid answers and save regardless
        invalid_count = len(df[
            df["model_answer"].isna() |
            (df["model_answer"] == "") |
            (df["model_answer"] == "Error") |
            (~df["model_answer"].apply(is_valid_answer))
        ])
        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries still have invalid answers after {max_retries} retries.")
            logging.warning(f"{invalid_count} entries still have invalid answers after {max_retries} retries.")
        else:
            print("All entries successfully populated with valid answers!")
            logging.info("All entries successfully populated with valid answers!")

        df.to_pickle(output_filename)
        print(f"Completed and saved to {output_filename} with {len(df)} rows!")
        logging.info(f"Completed and saved to {output_filename} with {len(df)} rows!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"An error occurred: {str(e)}")
        print("Please check your model, tokenizer, or environment.")

    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()