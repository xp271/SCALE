import torch
import pandas as pd
import re
# import config
from tqdm import tqdm
import time
import argparse
import logging
import os
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pickle
import numpy as np
from pathlib import Path

from mcq_option_utils import (
    build_prompt_suffix_cot,
    build_prompt_suffix_logit_only,
    option_letters_and_char_starts,
)

# Set up logging
logging.basicConfig(filename='inference.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaMA inference with chain-of-thought and/or logit computation using Transformers.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path (e.g. fake_quant_model dir)")
    parser.add_argument("--base_model_name", type=str, default=None, help="Base HF model when loading from fake_quant dir")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu", "commonsenseqa"],
        help="Used for output path only (output_inference/{dataset}/...). Data format is inferred from full_question.",
    )
    parser.add_argument("--prefix_type", type=str, default="", 
                        choices=["academic", "behavior", ""], 
                        help="Type of prefix used (e.g., 'academic', 'behavior').")
    parser.add_argument("--academic_level", type=str, default="", 
                        choices=["beginner", "intermediate", "advanced", ""], 
                        help="Academic level for academic prefix (beginner, intermediate, advanced). "
                             "Only applies when prefix_type='academic'.")
    parser.add_argument("--prefix_subtype", type=str, default="", 
                        choices=["original", "mixing_subject", "third_pov", ""], 
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
    parser.add_argument("--inference_mode", type=str, default="logit_and_cot", 
                        choices=["logit_only", "logit_and_cot"], 
                        help="Inference mode: 'logit_only' for logit-based answer selection without CoT, "
                             "'logit_and_cot' for chain-of-thought generation and logit-based selection.")
    parser.add_argument("--inference_layer", type=str, default="last", 
                        choices=["", "all", "odd", "even", "last"], 
                        help="Layers to compute logits: '' or 'all' for all layers, 'odd' for odd-numbered layers (including last), "
                             "'even' for even-numbered layers (including last), 'last' for last layer only.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for invalid answers")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cuda:N', or 'cpu'. Use e.g. 'cuda:1' to avoid OOM on GPU 0.")
    parser.add_argument(
        "--save_option_hs",
        action="store_true",
        help="Also save per-layer hidden states at tokenizer positions for option label tokens "
             "(lines like 'A. ...', 'E. ...' in full_question). Increases pickle size.",
    )
    parser.add_argument(
        "--stream_hs_dir",
        type=str,
        default=None,
        help="If set, after each row write layer_hidden_states + layer_option_hidden_states to one .pkl file "
             "under this directory and leave those DataFrame cells as None (saves CPU RAM; main pkl stays small). "
             "Does not reduce GPU peak memory during forward (output_hidden_states still materialized).",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="output_inference",
        help=(
            "Output root directory for pkl files (relative to cwd or absolute). "
            "Final path is {output_base}/{dataset}/{question_type}[/{prefix_type}/{prefix_subtype}/{academic_level}]/{basename}_{mode}_{layer}[_{seed}].pkl. "
            "Default: 'output_inference' (preserves the historical layout)."
        ),
    )
    return parser.parse_args()

def is_valid_answer(answer):
    """Check if the answer is a single uppercase letter."""
    return isinstance(answer, str) and len(answer) == 1 and answer.isupper() and answer.isalpha()

PROMPT_Q_PREFIX = "Question: ||"


def build_prompt(question: str, inference_mode: str, letters: list[str]) -> str:
    if inference_mode == "logit_and_cot":
        return PROMPT_Q_PREFIX + question + build_prompt_suffix_cot(letters)
    return PROMPT_Q_PREFIX + question + build_prompt_suffix_logit_only(letters)


def _spill_hs_row_to_disk(
    stream_dir: Path,
    row_idx,
    data_seed: int | None,
    layer_hidden_states: dict,
    layer_option_hidden_states: dict,
) -> str | None:
    """Pickle HS dicts for one row; return absolute path, or None if nothing to write."""
    if not layer_hidden_states and not layer_option_hidden_states:
        return None
    stream_dir.mkdir(parents=True, exist_ok=True)
    sid = data_seed if data_seed is not None else "ns"
    safe_idx = str(row_idx).replace(os.sep, "_").replace("/", "_")
    path = stream_dir / f"hs_{sid}_{safe_idx}.pkl"
    payload = {
        "layer_hidden_states": layer_hidden_states or {},
        "layer_option_hidden_states": layer_option_hidden_states or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return str(path.resolve())


def get_layer_indices(total_layers, inference_layer):
    """Determine which layer indices to process based on inference_layer argument."""
    if inference_layer in ["", "all"]:
        return list(range(total_layers))
    elif inference_layer == "odd":
        return [i for i in range(total_layers) if i % 2 == 1 or i == total_layers - 1]
    elif inference_layer == "even":
        return [i for i in range(total_layers) if i % 2 == 0 or i == total_layers - 1]
    elif inference_layer == "last":
        return [total_layers - 1]
    else:
        raise ValueError(f"Invalid inference_layer: {inference_layer}")

def process_question(
    question,
    tokenizer,
    model,
    inference_mode,
    inference_layer,
    question_index,
    save_option_hs: bool = False,
):
    try:
        logging.debug(f"Processing question at index {question_index}: '{question[:50]}...'")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at index {question_index}: must be a non-empty string, got '{question}'")

        letters, option_char = option_letters_and_char_starts(question)
        prompt = build_prompt(question, inference_mode, letters)
        question_start_in_prompt = len(PROMPT_Q_PREFIX)

        tok_batch = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=bool(save_option_hs),
        )
        offset_rows = tok_batch.get("offset_mapping")
        spans = offset_rows[0] if offset_rows is not None else None

        input_ids = tok_batch["input_ids"].to(model.device)
        attention_mask = tok_batch["attention_mask"].to(model.device)

        option_tok: dict[str, int | None] = {}
        if save_option_hs and option_char and spans is not None:
            for L, c_in_q in option_char.items():
                abs_c = question_start_in_prompt + c_in_q
                ti = None
                for tidx, span in enumerate(spans):
                    if span is None or span[0] is None:
                        continue
                    s, e = int(span[0]), int(span[1])
                    if s <= abs_c < e:
                        ti = tidx
                        break
                option_tok[L] = ti
            if any(v is None for v in option_tok.values()):
                logging.warning(
                    f"Could not map all option letters to tokens (option_tok={option_tok}); "
                    f"skip option HS for index {question_index}."
                )
                option_tok = {}
        elif save_option_hs and not option_char:
            logging.warning(
                f"No contiguous A./B./... option lines parsed in question (index {question_index}); skip option HS."
            )
        elif save_option_hs and spans is None:
            logging.warning(
                f"Tokenizer has no offset_mapping (index {question_index}); "
                f"use a fast tokenizer to enable --save_option_hs."
            )

        # Generate CoT output for logit_and_cot mode
        raw_output = ""
        if inference_mode == "logit_and_cot":
            logging.info(f"Generating CoT for index {question_index}: {prompt[:100]}...")
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
            decoded_output = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
            if "||" in prompt:
                parts = (prompt + decoded_output).split("||")
                raw_output = parts[2].lstrip() if len(parts) >= 3 else decoded_output
            else:
                raw_output = decoded_output
            logging.debug(f"Raw model output: {raw_output}")

        # Compute logits for answer selection（同时拿到所有层的 hidden_states）
        logging.info(f"Computing logits for index {question_index}: {prompt[:100]}...")
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of hidden states for each layer
            logits = outputs.logits[:, -1, :]  # Logits for the last token

        # Map answer tokens (per option letter)
        answer_tokens = {
            letter: tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters
        }
        space_answer_tokens = {
            letter: tokenizer.encode(f" {letter}", add_special_tokens=False)[0] for letter in letters
        }
        logging.debug(f"Answer token IDs: {answer_tokens}")
        logging.debug(f"Space-prefixed answer token IDs: {space_answer_tokens}")

        neg = float("-inf")
        init_logits = {L: neg for L in letters}

        # Initialize layer-wise logits storage
        total_layers = len(hidden_states) - 1  # Number of transformer layers
        layer_indices = get_layer_indices(total_layers, inference_layer)
        layer_logits = {f"layer_{i}": dict(init_logits) for i in layer_indices}
        # Initialize layer-wise hidden states storage（只保存最后一个 token 的向量）
        layer_hidden_states = {}
        layer_option_hidden_states: dict = {}
        if save_option_hs and option_tok:
            layer_option_hidden_states["_meta"] = {
                "option_char_in_question": dict(option_char),
                "option_token_idx": {k: int(option_tok[k]) for k in sorted(option_tok) if option_tok[k] is not None},
            }

        # Process logits for each specified layer
        for layer_idx in layer_indices:
            # Get hidden states for the layer (last token)
            hidden_state = hidden_states[layer_idx + 1][:, -1, :]  # +1 because hidden_states includes input embeddings
            # Project hidden state to logits using the model's language model head
            layer_logits_raw = model.lm_head(hidden_state)
            answer_logits = layer_logits[f"layer_{layer_idx}"]

            # Save hidden state vector (to CPU, float16 numpy)
            try:
                hs_vec = hidden_state.detach().cpu().to(torch.float16).numpy()
            except Exception:
                hs_vec = hidden_state.detach().cpu().numpy()
            layer_hidden_states[f"layer_{layer_idx}"] = hs_vec

            if option_tok:
                per_letter = {}
                hs_layer = hidden_states[layer_idx + 1]
                seq_len = hs_layer.shape[1]
                for L, ti in option_tok.items():
                    if ti is None or ti < 0 or ti >= seq_len:
                        continue
                    h_opt = hs_layer[:, ti, :]
                    try:
                        per_letter[L] = h_opt.detach().cpu().to(torch.float16).numpy()
                    except Exception:
                        per_letter[L] = h_opt.detach().cpu().numpy()
                if per_letter:
                    layer_option_hidden_states[f"layer_{layer_idx}"] = per_letter

            for letter in letters:
                token_id = answer_tokens[letter]
                space_token_id = space_answer_tokens[letter]
                logit = max(
                    layer_logits_raw[0, token_id].item(),
                    layer_logits_raw[0, space_token_id].item()
                )
                answer_logits[letter] = logit

            logging.debug(f"Layer {layer_idx} logits for {letters}: {answer_logits}")

        # Use last layer logits for answer selection
        last_layer_logits = layer_logits[f"layer_{total_layers-1}"]
        answer_probs = {}
        for letter, logprob in last_layer_logits.items():
            answer_probs[letter] = torch.exp(torch.tensor(logprob)).item() if logprob != float('-inf') else 0.0

        total_prob = sum(answer_probs.values())
        if total_prob > 0:
            answer_probs = {letter: prob / total_prob for letter, prob in answer_probs.items()}
        else:
            uni = 1.0 / len(letters) if letters else 0.25
            answer_probs = {letter: uni for letter in letters}

        selected_answer = max(answer_probs, key=answer_probs.get)
        logging.debug(f"Answer based on probabilities: {selected_answer}, Probabilities: {answer_probs}")

        if not is_valid_answer(selected_answer):
            logging.warning(f"Invalid answer at index {question_index}: '{selected_answer}' for question: '{question[:50]}...'")
            return "Error", layer_logits, raw_output, layer_hidden_states, layer_option_hidden_states

        return selected_answer, layer_logits, raw_output, layer_hidden_states, layer_option_hidden_states

    except Exception as e:
        logging.error(f"Error processing question at index {question_index} '{question[:50]}...': {str(e)}\n{traceback.format_exc()}")
        return "Error", {}, "Error in processing", {}, {}

def main():
    args = parse_args()
    model_name = args.model_name
    dataset = args.dataset
    prefix_type = args.prefix_type
    academic_level = args.academic_level
    prefix_subtype = args.prefix_subtype
    question_type = args.question_type
    input_filename = args.input_filename
    data_seed = getattr(args, "data_seed", None)
    if data_seed is not None:
        base, ext = os.path.splitext(input_filename)
        base = re.sub(r"_\d+$", "", base)
        input_filename = f"{base}_{data_seed}{ext}"
        print(f"Using seed {data_seed} dataset: {input_filename}")
    full_question_column = args.full_question_column
    inference_mode = args.inference_mode
    inference_layer = args.inference_layer
    max_retries = args.max_retries
    device_arg = getattr(args, "device", "auto")
    save_option_hs = getattr(args, "save_option_hs", False)
    stream_hs_dir_raw = getattr(args, "stream_hs_dir", None)
    stream_hs_root: Path | None = (
        Path(stream_hs_dir_raw).expanduser().resolve()
        if stream_hs_dir_raw
        else None
    )
    if stream_hs_root is not None:
        stream_hs_root.mkdir(parents=True, exist_ok=True)
        print(f"Streaming HS payloads to {stream_hs_root} (DataFrame stores paths only).")
        logging.info(f"Streaming HS payloads to {stream_hs_root}")

    # Validation
    if academic_level and prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")
    if question_type == "prefix_and_opinion" and not prefix_type:
        raise ValueError("For 'prefix_and_opinion' question_type, a prefix_type (e.g., 'academic' or 'behavior') must be specified.")

    import sys
    from pathlib import Path
    _script_dir = Path(__file__).resolve().parent
    _repo_root = _script_dir.parent.parent  # experiments -> LLM-sycophancy
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    if not hf_token:
        raise ValueError(
            "HF token not set. Export HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), or set syco.huggingface_token in "
            "config/pipeline_config.yaml when using run_pipeline (subprocess inherits HF_TOKEN)."
        )

    try:
        base_model_name = getattr(args, "base_model_name", None)
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
        load_on_specific_device = (
            isinstance(device_arg, str) and device_arg.startswith("cuda:") and device_arg != "cuda"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else torch.device(device_arg)

        if use_fake_quant_loader:
            from utils.standalone_fake_quant import load_model_for_fake_quant
            print("Loading tokenizer and model (fake_quant path)...")
            logging.info("Loading tokenizer and model (fake_quant path)...")
            model, tokenizer = load_model_for_fake_quant(
                model_name,
                base_model_name,
                device=device if not load_on_specific_device else None,
                device_map=device_arg if load_on_specific_device else None,
                token=hf_token,
            )
            if not load_on_specific_device:
                model = model.to(device)
            model.eval()
        else:
            print("Loading tokenizer...")
            logging.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
            )

            print("Loading Transformers model...")
            logging.info("Loading Transformers model...")
            config = AutoConfig.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                force_download=False,
                resume_download=True,
            )
            print(f"Model configuration: {config}")
            logging.info(f"Model configuration: {config}")

            from transformers import modeling_utils
            if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
                modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
            if not hasattr(config, 'parallel_style') or config.parallel_style is None:
                config.parallel_style = "none"
                logging.warning("Patched config.parallel_style to 'none'.")
            if not hasattr(config, '_fsdp_config') or config._fsdp_config is None:
                config._fsdp_config = {}
                logging.warning("Patched config._fsdp_config to empty dict.")
            if not hasattr(config, 'model_parallel') or config.model_parallel is None:
                config.model_parallel = False
                logging.warning("Patched config.model_parallel to False.")

            model_kwargs = dict(
                config=config,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                force_download=False,
                resume_download=True,
            )
            if load_on_specific_device:
                model_kwargs["device_map"] = device_arg
            else:
                model_kwargs["device_map"] = None
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if not load_on_specific_device:
                model = model.to(device)
            model.eval()

        print(f"Loading DataFrame from {input_filename}...")
        logging.info(f"Loading DataFrame from {input_filename}...")
        df = pd.read_pickle(input_filename)
        print(f"Loaded DataFrame with {len(df)} entries.")
        logging.info(f"Loaded DataFrame with {len(df)} entries.")
        print(f"DataFrame columns: {df.columns}")
        logging.info(f"DataFrame columns: {df.columns}")

        if full_question_column not in df.columns:
            raise ValueError(f"Input DataFrame must contain a '{full_question_column}' column.")

        print(f"Validating '{full_question_column}' column...")
        logging.info(f"Validating '{full_question_column}' column...")
        invalid_questions = df[full_question_column].apply(lambda x: not isinstance(x, str) or not x.strip())
        if invalid_questions.any():
            invalid_indices = invalid_questions[invalid_questions].index.tolist()
            invalid_samples = df.loc[invalid_indices, full_question_column].head().to_dict()
            raise ValueError(f"Found {len(invalid_indices)} invalid questions in '{full_question_column}' column: {invalid_samples}")

        # Initialize DataFrame columns
        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "layer_logits" not in df.columns:
            df["layer_logits"] = None
        if "raw_output" not in df.columns:
            df["raw_output"] = None
        if "layer_hidden_states" not in df.columns:
            df["layer_hidden_states"] = None
        if "layer_option_hidden_states" not in df.columns:
            df["layer_option_hidden_states"] = None
        if "hs_artifact_path" not in df.columns:
            df["hs_artifact_path"] = None

        def _assign_hs_columns(idx, lhs, lohs):
            if stream_hs_root is not None:
                p = _spill_hs_row_to_disk(stream_hs_root, idx, data_seed, lhs, lohs)
                df.at[idx, "hs_artifact_path"] = p
                df.at[idx, "layer_hidden_states"] = None
                df.at[idx, "layer_option_hidden_states"] = None
            else:
                df.at[idx, "layer_hidden_states"] = lhs
                df.at[idx, "layer_option_hidden_states"] = lohs

        print("Testing with first 5 questions...")
        logging.info("Testing with first 5 questions...")
        for idx in df.index[:5]:
            question = df.at[idx, full_question_column]
            answer, layer_logits, raw_out, layer_hidden_states, layer_option_hidden_states = process_question(
                question, tokenizer, model, inference_mode, inference_layer, idx,
                save_option_hs=save_option_hs,
            )
            print(f"Test question at index {idx}: Answer = {answer}, Layer Logits = {layer_logits}, Raw Output = {raw_out[:100]}...")
            logging.info(f"Test question at index {idx}: Answer = {answer}, Layer Logits = {layer_logits}, Raw Output = {raw_out}")
            _assign_hs_columns(idx, layer_hidden_states, layer_option_hidden_states)
            del layer_hidden_states, layer_option_hidden_states
            torch.cuda.empty_cache()

        print("Processing all questions...")
        logging.info("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]):
                question = df.at[idx, full_question_column]
                answer, layer_logits, raw_out, layer_hidden_states, layer_option_hidden_states = process_question(
                    question, tokenizer, model, inference_mode, inference_layer, idx,
                    save_option_hs=save_option_hs,
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_out
                _assign_hs_columns(idx, layer_hidden_states, layer_option_hidden_states)
                del layer_hidden_states, layer_option_hidden_states
                torch.cuda.empty_cache()

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
            for idx in tqdm(invalid_indices, desc=f"Retry {retry_count + 1}"):
                question = df.at[idx, full_question_column]
                answer, layer_logits, raw_out, layer_hidden_states, layer_option_hidden_states = process_question(
                    question, tokenizer, model, inference_mode, inference_layer, idx,
                    save_option_hs=save_option_hs,
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_out
                _assign_hs_columns(idx, layer_hidden_states, layer_option_hidden_states)
                del layer_hidden_states, layer_option_hidden_states
                torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        # Construct output directory: {output_base}/{dataset}/{question_type}/{prefix_type}/{prefix_subtype}/{academic_level}
        _output_base = getattr(args, "output_base", None) or "output_inference"
        output_dir_parts = [f"{_output_base}/{dataset}"]
        if question_type:
            output_dir_parts.append(question_type)
        if prefix_type:
            output_dir_parts.append(prefix_type)
            output_dir_parts.append(prefix_subtype)
            if prefix_type == "academic":
                output_dir_parts.append(academic_level)
        output_dir = os.path.join(*[part for part in output_dir_parts if part])

        os.makedirs(output_dir, exist_ok=True)

        _bn = getattr(args, "model_output_name", None) or model_name.split("/")[-1].replace(".", "_")
        output_basename = _bn.replace("\\", "_").replace("/", "_")
        inference_mode_str = 'cot' if inference_mode == 'logit_and_cot' else 'logit'
        seed_suffix = f"_{data_seed}" if data_seed is not None else ""
        output_filename = f"{output_dir}/{output_basename}_{inference_mode_str}_{inference_layer}{seed_suffix}.pkl"

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

        print(f"Saving to {output_filename}...")
        logging.info(f"Saving to {output_filename}...")
        df.to_pickle(output_filename)
        print(f"Completed and saved to {output_filename} with {len(df)} rows!")
        logging.info(f"Completed and saved to {output_filename} with {len(df)} rows!")

    except Exception as e:
        print(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        logging.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        raise

    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()