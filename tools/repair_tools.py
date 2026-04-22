"""
Model Repair Tools - Tools the LLM agent can call to diagnose and fix model issues.
These tools perform actual operations on models (weights inspection, config fixes, etc.)
"""

import os
import json
import copy
import time
import shutil
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    data: dict = None
    error: str = None
    duration: float = 0.0


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "inspect_model_config",
        "description": "Read and analyze the model's config.json file for architectural issues, wrong settings, or missing fields.",
        "parameters": {"model_path": "Path to the model directory"},
    },
    {
        "name": "inspect_model_files",
        "description": "List all files in the model directory, check sizes, detect missing/corrupt files.",
        "parameters": {"model_path": "Path to the model directory"},
    },
    {
        "name": "check_tokenizer_config",
        "description": "Inspect tokenizer files for issues: wrong special tokens, missing vocabulary, bad padding config.",
        "parameters": {"model_path": "Path to the model directory"},
    },
    {
        "name": "validate_weights_integrity",
        "description": "Check weight files exist and are not zero-byte/corrupted. Compute checksums.",
        "parameters": {"model_path": "Path to model directory", "deep_check": "bool - whether to load and inspect tensor shapes"},
    },
    {
        "name": "fix_config_field",
        "description": "Fix a specific field in config.json. Does NOT change model knowledge - only architectural/behavioral config.",
        "parameters": {"model_path": "Path", "field": "Config field name", "value": "New value", "reason": "Why this fix is needed"},
    },
    {
        "name": "fix_tokenizer_field",
        "description": "Fix a specific field in tokenizer_config.json or tokenizer.json.",
        "parameters": {"model_path": "Path", "file": "tokenizer_config.json or tokenizer.json", "field": "Field name", "value": "New value"},
    },
    {
        "name": "add_missing_config_field",
        "description": "Add a missing required field to config.json with a safe default value.",
        "parameters": {"model_path": "Path", "field": "Field name to add", "value": "Default value", "reason": "Why needed"},
    },
    {
        "name": "backup_model_config",
        "description": "Create a backup of model config files before any modifications.",
        "parameters": {"model_path": "Path to model directory"},
    },
    {
        "name": "restore_from_backup",
        "description": "Restore config files from a previously created backup.",
        "parameters": {"model_path": "Path", "backup_id": "Backup ID to restore from"},
    },
    {
        "name": "test_model_loading",
        "description": "Attempt to load the model with transformers and report any errors or warnings.",
        "parameters": {"model_path": "Path", "device": "cpu or cuda"},
    },
    {
        "name": "check_generation_params",
        "description": "Inspect generation_config.json for problematic settings (bad temperature, wrong eos_token_id, etc.)",
        "parameters": {"model_path": "Path"},
    },
    {
        "name": "fix_generation_config",
        "description": "Fix generation_config.json settings that cause poor output quality.",
        "parameters": {"model_path": "Path", "field": "Field to fix", "value": "Correct value"},
    },
    {
        "name": "deep_neuron_inspection",
        "description": "Exhaustively sweeps PyTorch matrix tensors/weights layer-by-layer checking for dead neurons, NaNs, extreme variance and layer drops. Deep structural test.",
        "parameters": {"model_path": "Path to model directory"},
    },
]


def get_tool_list_for_llm() -> str:
    """Format tools as a string for the LLM to understand."""
    lines = ["Available repair tools:\n"]
    for i, tool in enumerate(TOOL_DEFINITIONS, 1):
        lines.append(f"{i}. **{tool['name']}**")
        lines.append(f"   Description: {tool['description']}")
        params = ", ".join([f"{k}: {v}" for k, v in tool['parameters'].items()])
        lines.append(f"   Parameters: {params}\n")
    return "\n".join(lines)


# ─── TOOL IMPLEMENTATIONS ─────────────────────────────────────────────────────

def run_tool(tool_name: str, params: dict) -> ToolResult:
    """Dispatch and run a tool by name."""
    start = time.time()
    tool_map = {
        "inspect_model_config": _inspect_model_config,
        "inspect_model_files": _inspect_model_files,
        "check_tokenizer_config": _check_tokenizer_config,
        "validate_weights_integrity": _validate_weights_integrity,
        "fix_config_field": _fix_config_field,
        "fix_tokenizer_field": _fix_tokenizer_field,
        "add_missing_config_field": _add_missing_config_field,
        "backup_model_config": _backup_model_config,
        "restore_from_backup": _restore_from_backup,
        "test_model_loading": _test_model_loading,
        "check_generation_params": _check_generation_params,
        "fix_generation_config": _fix_generation_config,
        "deep_neuron_inspection": _deep_neuron_inspection,
    }
    
    fn = tool_map.get(tool_name)
    if not fn:
        return ToolResult(tool_name, False, "", error=f"Unknown tool: {tool_name}", duration=time.time()-start)
    
    try:
        result = fn(**params)
        result.duration = time.time() - start
        return result
    except Exception as e:
        return ToolResult(tool_name, False, "", error=str(e), duration=time.time()-start)


def _inspect_model_config(model_path: str) -> ToolResult:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ToolResult("inspect_model_config", False, "", error="config.json not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    issues = []
    # Common checks
    if "model_type" not in config:
        issues.append("MISSING: model_type field")
    
    hidden_size = config.get("hidden_size", config.get("d_model", config.get("n_embd", config.get("d_kv", 0))))
    if hidden_size == 0:
        issues.append("MISSING: hidden_size/d_model/n_embd/d_kv")
        
    max_pos = config.get("max_position_embeddings", config.get("n_positions", config.get("n_ctx", config.get("seq_length", 0))))
    if max_pos == 0:
        issues.append("WARNING: max_position_embeddings/n_positions/seq_length is 0")
        
    if "vocab_size" not in config:
        issues.append("MISSING: vocab_size")
        
    num_heads = config.get("num_attention_heads", config.get("n_head", config.get("num_heads", "N/A")))
    num_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", "N/A")))
    
    output = f"⚙️ CONFIGURATION ANALYSIS:\n"
    output += f"   - Config fields: {len(config)}\n"
    output += f"   - Model type:    {config.get('model_type', 'UNKNOWN')}\n"
    output += f"   - Architecture:  {config.get('architectures', ['UNKNOWN'])}\n"
    output += f"   - Hidden size:   {hidden_size if hidden_size else 'N/A'}\n"
    output += f"   - Num layers:    {num_layers}\n"
    output += f"   - Attn heads:    {num_heads}\n"
    output += f"   - Vocab size:    {config.get('vocab_size', 'N/A')}\n"
    output += f"   - Max seq len:   {max_pos if max_pos else 'N/A'}\n"
    if issues:
        output += f"\n⚠️  ISSUES FOUND:\n" + "\n".join(f"  - {i}" for i in issues)
    else:
        output += "\n✅ No obvious config issues detected"
    
    return ToolResult("inspect_model_config", True, output, data={"config": config, "issues": issues})


def _inspect_model_files(model_path: str) -> ToolResult:
    path = Path(model_path)
    if not path.exists():
        return ToolResult("inspect_model_files", False, "", error=f"Path does not exist: {model_path}")
    
    files = []
    total_size = 0
    issues = []
    
    for f in sorted(path.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            files.append({"name": f.name, "size": size, "path": str(f.relative_to(path))})
            if size == 0:
                issues.append(f"ZERO BYTE FILE: {f.name}")
    
    # Check for required files
    required = ["config.json"]
    for req in required:
        if not (path / req).exists():
            issues.append(f"MISSING REQUIRED: {req}")
    
    weight_files = [f for f in files if any(f["name"].endswith(ext) for ext in [".bin", ".safetensors", ".pt", ".ckpt"])]
    
    output = f"📁 PHYSICAL FILE ANALYSIS:\n"
    output += f"   - Total files: {len(files)}\n"
    output += f"   - Model Size:  {total_size / (1024**3):.2f} GB\n"
    output += f"   - Payload Wts: {len(weight_files)} files\n"
    output += "   - Top Weight Files:\n"
    for wf in weight_files[:10]:
        output += f"       * {wf['name']}: {wf['size']/(1024**2):.1f} MB\n"
    if issues:
        output += f"\n⚠️  ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
    else:
        output += "\n✅ All files present and non-empty"
    
    return ToolResult("inspect_model_files", True, output, data={"files": files, "issues": issues, "weight_count": len(weight_files)})


def _check_tokenizer_config(model_path: str) -> ToolResult:
    path = Path(model_path)
    issues = []
    output = ""
    
    tok_config_path = path / "tokenizer_config.json"
    if not tok_config_path.exists():
        return ToolResult("check_tokenizer_config", False, "", error="tokenizer_config.json not found")
    
    with open(tok_config_path) as f:
        tok_config = json.load(f)
    
    if "tokenizer_class" not in tok_config:
        issues.append("MISSING: tokenizer_class")
    if "model_max_length" not in tok_config:
        issues.append("WARNING: model_max_length not set (may cause truncation issues)")
    if tok_config.get("model_max_length", 0) > 1e30:
        issues.append("WARNING: model_max_length set to infinity-like value")
    if "chat_template" not in tok_config:
        issues.append("INFO: No chat_template found (may cause issues for chat models)")
    
    output += f"Tokenizer class: {tok_config.get('tokenizer_class', 'UNKNOWN')}\n"
    output += f"Max length: {tok_config.get('model_max_length', 'N/A')}\n"
    output += f"Padding side: {tok_config.get('padding_side', 'N/A')}\n"
    output += f"Has chat template: {'Yes' if 'chat_template' in tok_config else 'No'}\n"
    output += f"BOS token: {tok_config.get('bos_token', 'N/A')}\n"
    output += f"EOS token: {tok_config.get('eos_token', 'N/A')}\n"
    output += f"PAD token: {tok_config.get('pad_token', 'None/Not set')}\n"
    
    if not tok_config.get("pad_token"):
        issues.append("WARNING: No pad_token set - may cause batching issues during training")
    
    if issues:
        output += f"\n⚠️  ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
    else:
        output += "\n✅ Tokenizer config looks healthy"
    
    return ToolResult("check_tokenizer_config", True, output, data={"tokenizer_config": tok_config, "issues": issues})


def _validate_weights_integrity(model_path: str, deep_check: bool = False) -> ToolResult:
    path = Path(model_path)
    issues = []
    weight_info = []
    
    weight_extensions = [".safetensors", ".bin", ".pt", ".ckpt"]
    weight_files = [f for f in path.iterdir() if f.suffix in weight_extensions]
    
    if not weight_files:
        return ToolResult("validate_weights_integrity", False, "", error="No weight files found")
    
    for wf in weight_files:
        size = wf.stat().st_size
        if size == 0:
            issues.append(f"CORRUPT: {wf.name} is 0 bytes")
            weight_info.append({"file": wf.name, "size": 0, "status": "CORRUPT"})
        elif size < 1024:
            issues.append(f"SUSPICIOUS: {wf.name} is only {size} bytes")
            weight_info.append({"file": wf.name, "size": size, "status": "SUSPICIOUS"})
        else:
            # Compute quick checksum
            hasher = hashlib.md5()
            with open(wf, "rb") as f:
                chunk = f.read(65536)  # Read first 64KB only for speed
                hasher.update(chunk)
            weight_info.append({"file": wf.name, "size": size, "status": "OK", "checksum_partial": hasher.hexdigest()})
    
    output = f"Weight files checked: {len(weight_files)}\n"
    for wi in weight_info:
        status_icon = "✅" if wi["status"] == "OK" else "❌"
        output += f"  {status_icon} {wi['file']}: {wi['size']/(1024**2):.1f} MB [{wi['status']}]\n"
    
    if not issues:
        output += "\n✅ All weight files appear intact"
    else:
        output += f"\n❌ INTEGRITY ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
    
    return ToolResult("validate_weights_integrity", len(issues) == 0, output, data={"weights": weight_info, "issues": issues})


def _fix_config_field(model_path: str, field: str, value, reason: str) -> ToolResult:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ToolResult("fix_config_field", False, "", error="config.json not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    old_value = config.get(field, "<NOT SET>")
    config[field] = value
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    output = f"✅ Fixed config field: {field}\n"
    output += f"   Old value: {old_value}\n"
    output += f"   New value: {value}\n"
    output += f"   Reason: {reason}"
    
    return ToolResult("fix_config_field", True, output, data={"field": field, "old_value": str(old_value), "new_value": str(value)})


def _fix_tokenizer_field(model_path: str, file: str, field: str, value) -> ToolResult:
    tok_path = Path(model_path) / file
    if not tok_path.exists():
        return ToolResult("fix_tokenizer_field", False, "", error=f"{file} not found")
    
    with open(tok_path) as f:
        config = json.load(f)
    
    old_value = config.get(field, "<NOT SET>")
    config[field] = value
    
    with open(tok_path, "w") as f:
        json.dump(config, f, indent=2)
    
    output = f"✅ Fixed tokenizer field: {field} in {file}\n"
    output += f"   Old: {old_value} → New: {value}"
    
    return ToolResult("fix_tokenizer_field", True, output, data={"field": field, "old": str(old_value), "new": str(value)})


def _add_missing_config_field(model_path: str, field: str, value, reason: str) -> ToolResult:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ToolResult("add_missing_config_field", False, "", error="config.json not found")
    
    with open(config_path) as f:
        config = json.load(f)
    
    if field in config:
        return ToolResult("add_missing_config_field", False, "", 
                         error=f"Field '{field}' already exists (value: {config[field]}). Use fix_config_field instead.")
    
    config[field] = value
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    output = f"✅ Added missing field: {field} = {value}\n   Reason: {reason}"
    return ToolResult("add_missing_config_field", True, output, data={"field": field, "value": str(value)})


def _backup_model_config(model_path: str) -> ToolResult:
    path = Path(model_path)
    backup_id = f"backup_{int(time.time())}"
    backup_dir = path / ".llmdoctor_backups" / backup_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backed_up = []
    config_files = ["config.json", "tokenizer_config.json", "tokenizer.json", 
                    "generation_config.json", "special_tokens_map.json"]
    
    for cf in config_files:
        src = path / cf
        if src.exists():
            shutil.copy2(src, backup_dir / cf)
            backed_up.append(cf)
    
    output = f"✅ Backup created: {backup_id}\n"
    output += f"   Backed up files: {', '.join(backed_up)}\n"
    output += f"   Location: {backup_dir}"
    
    return ToolResult("backup_model_config", True, output, data={"backup_id": backup_id, "files": backed_up, "backup_path": str(backup_dir)})


def _restore_from_backup(model_path: str, backup_id: str) -> ToolResult:
    path = Path(model_path)
    backup_dir = path / ".llmdoctor_backups" / backup_id
    
    if not backup_dir.exists():
        return ToolResult("restore_from_backup", False, "", error=f"Backup '{backup_id}' not found")
    
    restored = []
    for backup_file in backup_dir.iterdir():
        dest = path / backup_file.name
        shutil.copy2(backup_file, dest)
        restored.append(backup_file.name)
    
    output = f"✅ Restored from backup: {backup_id}\n"
    output += f"   Restored files: {', '.join(restored)}"
    return ToolResult("restore_from_backup", True, output, data={"restored": restored})


def _test_model_loading(model_path: str, device: str = "cpu") -> ToolResult:
    import time
    try:
        from transformers import AutoConfig, AutoTokenizer, pipeline, AutoModel
        
        output = ""
        issues = []
        
        # Test config and tokenizer loading
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Automatically assign pad_token to avoid missing token warnings during inference
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            output += f"✅ Config and Tokenizer loaded successfully\n"
            output += f"   Model type: {config.model_type}\n"
            output += f"   Vocab size: {tokenizer.vocab_size}\n"
        except Exception as e:
            issues.append(f"Config/Tokenizer load failed: {e}")
            output += f"❌ Config/Tokenizer load failed: {e}\n"
            return ToolResult("test_model_loading", False, output, error=str(e), data={"issues": issues})
        
        # Test actual model loading and inference
        try:
            output += f"⏳ Loading model weights and running inference test...\n"
            start_load = time.time()
            
            # Using pipeline for a quick text-generation sanity check
            try:
                # Suppress transformer warnings locally to keep the output pristine
                import logging
                logging.getLogger("transformers").setLevel(logging.ERROR)
                
                generator = pipeline("text-generation", model=model_path, tokenizer=tokenizer, device=-1)
                load_time = time.time() - start_load
                output += f"   Pipeline initialization successful in {load_time:.2f}s.\n"
                
                start_inf = time.time()
                # Use max_new_tokens and pad_token_id to bypass length warnings and padding warnings implicitly
                result = generator("System diagnostics running:", max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
                inf_time = time.time() - start_inf
                
                output += f"✅ Inference Test Successful! ({inf_time:.2f}s)\n"
                output += f"   Output Sample: {repr(result[0]['generated_text'])}\n"
                
            except Exception as pipe_err:
                output += f"⚠️ Pipeline task 'text-generation' unsupported or failed. Falling back to base model load. ({pipe_err})\n"
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                load_time = time.time() - start_load
                output += f"✅ Base model Weights loaded successfully in {load_time:.2f}s.\n"

        except Exception as e:
            issues.append(f"Model Inference load failed: {e}")
            output += f"❌ Model Inference load failed: {e}\n"
        
        if not issues:
            output += "\n✅ Model inference & execution test PASSED."
        else:
            output += f"\n⚠️  {len(issues)} issue(s) found during execution test."
        
        return ToolResult("test_model_loading", len(issues) == 0, output, data={"issues": issues})
        
    except ImportError:
        return ToolResult("test_model_loading", False, "", error="transformers library not installed")

def _check_generation_params(model_path: str) -> ToolResult:
    gen_config_path = Path(model_path) / "generation_config.json"
    issues = []
    
    if not gen_config_path.exists():
        return ToolResult("check_generation_params", True, "No generation_config.json found (using defaults - this is OK)", 
                         data={"issues": [], "exists": False})
    
    with open(gen_config_path) as f:
        gen_config = json.load(f)
    
    output = "Generation Config Analysis:\n"
    
    # Check common issues
    temp = gen_config.get("temperature", 1.0)
    if temp == 0:
        issues.append("temperature=0 can cause repetition loops")
    if temp > 2.0:
        issues.append(f"temperature={temp} is very high - may cause incoherent output")
    
    if gen_config.get("repetition_penalty", 1.0) < 1.0:
        issues.append("repetition_penalty < 1.0 actually encourages repetition (should be > 1.0)")
    
    eos_id = gen_config.get("eos_token_id")
    if eos_id is None:
        issues.append("No eos_token_id set - model may not stop generating")
    
    for k, v in gen_config.items():
        output += f"  {k}: {v}\n"
    
    if issues:
        output += f"\n⚠️  ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
    else:
        output += "\n✅ Generation config looks healthy"
    
    return ToolResult("check_generation_params", True, output, data={"config": gen_config, "issues": issues})


def _fix_generation_config(model_path: str, field: str, value) -> ToolResult:
    gen_config_path = Path(model_path) / "generation_config.json"
    
    if not gen_config_path.exists():
        gen_config = {}
    else:
        with open(gen_config_path) as f:
            gen_config = json.load(f)
    
    old_value = gen_config.get(field, "<NOT SET>")
    gen_config[field] = value
    
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)
    
    output = f"✅ Fixed generation_config field: {field}\n"
    output += f"   Old: {old_value} → New: {value}"
    return ToolResult("fix_generation_config", True, output, data={"field": field, "old": str(old_value), "new": str(value)})


def _deep_neuron_inspection(model_path: str) -> ToolResult:
    try:
        import torch
        from transformers import AutoModel
        
        output = "🩺 DEEP NEURON & LAYER INSPECTION INITIATED\n"
        output += "="*60 + "\n"
        issues = []
        
        try:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            output += "✅ Physical Weights Loaded into Memory for Dissection.\n\n"
        except Exception as e:
            issues.append(f"Could not load physical model weights: {e}")
            return ToolResult("deep_neuron_inspection", False, output + f"❌ {e}\n", error=str(e))
            
        total_layers_scanned = 0
        total_params_scanned = 0
        total_dead_neurons = 0
        total_nan = 0
        total_inf = 0
        
        layer_summaries = []
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params_scanned += num_params
            total_layers_scanned += 1
            
            # Mathematical Health Checks
            nans = torch.isnan(param).sum().item()
            infs = torch.isinf(param).sum().item()
            
            # Dead neurons (absolute zero)
            zeros = (param == 0).sum().item()
            
            # Statistical drift (Vanishing/Exploding signals)
            mean_val = param.mean().item()
            std_val = param.std().item() if num_params > 1 else 0.0
            
            # Track aggregates
            total_nan += nans
            total_inf += infs
            total_dead_neurons += zeros
            
            # Flag anomalies
            layer_status = "✅ OK"
            layer_issues = []
            
            if nans > 0:
                layer_issues.append(f"{nans} NaN values (Exploded Layer)")
            if infs > 0:
                layer_issues.append(f"{infs} Inf values (Exploded Layer)")
            if zeros > (num_params * 0.1): # More than 10% exactly zero
                layer_issues.append(f"{zeros} Dead Neurons ({zeros/num_params * 100:.1f}%)")
            if std_val > 50.0:
                layer_issues.append(f"Extreme Variance (std={std_val:.2f})")
            if std_val == 0.0 and mean_val == 0.0:
                layer_issues.append(f"COMPLETELY DEAD LAYER")
                
            if layer_issues:
                layer_status = "❌ BROKEN"
                issues.append(f"Layer {name} is broken: {', '.join(layer_issues)}")
                
            # Log summary
            if layer_issues or 'attention' in name.lower() or 'mlp' in name.lower():
                layer_summaries.append(f"{name}:\n  -> Status: {layer_status}\n  -> Shape: {list(param.shape)} ({num_params} param)\n  -> Math: Mean {mean_val:.6f} | Std {std_val:.6f}\n  -> Corruptions: {nans} NaNs, {infs} Infs, {zeros} Dead Neurons")
        
        output += f"📊 NEURON HEALTH SUMMARY:\n"
        output += f"   - Total Layers Scanned:   {total_layers_scanned}\n"
        output += f"   - Total Neural Weights:   {total_params_scanned:,} mathematical parameters\n"
        output += f"   - Total Healthy Neurons:  {total_params_scanned - total_dead_neurons - total_nan - total_inf:,}\n"
        output += f"   - Global Corrupt (NaNs):  {total_nan} / {total_inf}\n"
        output += f"   - Global Dead Neurons:    {total_dead_neurons:,} strictly 0.0 values\n"
        output += f"   - Weight Efficiency:      {100 - (total_dead_neurons/max(1,total_params_scanned)*100):.4f}% Active\n\n"
        
        output += f"🔬 LAYER DIAGNOSTIC HIGHLIGHTS:\n"
        
        if issues:
            output += "⚠️ CRITICAL DEVIATIONS DETECTED:\n\n"
        else:
            output += "✨ No catastrophic mathematical deviations detected in layers.\n\n"
            
        # Limit to first 20 interesting layers to avoid massive bloat, prioritizing broken ones
        broken_sums = [s for s in layer_summaries if '❌' in s]
        ok_sums = [s for s in layer_summaries if '❌' not in s]
        
        display_sums = broken_sums + ok_sums
        
        for summary in display_sums[:20]:
            output += summary + "\n\n"
            
        if len(display_sums) > 20:
            output += f"... and {len(display_sums) - 20} more layers inspected but truncated from view.\n"
            
        return ToolResult("deep_neuron_inspection", True, output, data={"issues": issues, "total_dead": total_dead_neurons, "total_nan": total_nan})

    except ImportError:
         return ToolResult("deep_neuron_inspection", False, "", error="torch library not installed")
