"""
Diagnostic Engine - Interviews the user, generates questions, and diagnoses LLM issues.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from core.llm_client import get_completion, get_json_completion, stream_completion
from tools.repair_tools import run_tool


DIAGNOSTICIAN_SYSTEM = """You are an expert LLM diagnostician and AI systems doctor. 
Your job is to deeply understand problems with language models, training pipelines, 
inference systems, and AI applications. You ask precise, technical questions to uncover 
root causes. You think like a doctor: systematic, thorough, evidence-based.

When generating diagnostic questions, focus on:
- Model behavior symptoms (hallucinations, refusals, performance drops)
- Training issues (overfitting, underfitting, data quality, loss curves)
- Inference issues (latency, memory, batching, quantization artifacts)
- Data pipeline issues (preprocessing, tokenization, distribution shift)
- Architecture issues (attention patterns, layer problems, gradient flow)
- Deployment issues (hardware, serving, concurrency)"""


@dataclass
class DiagnosticSession:
    problem_description: str = ""
    model_path: str = ""
    timestamp: float = field(default_factory=time.time)
    qa_pairs: list = field(default_factory=list)  # Legacy QA, kept for loaded sessions
    automated_checks: dict = field(default_factory=dict) # {"tool_name": "output"}
    identified_issues: list = field(default_factory=list)
    severity_map: dict = field(default_factory=dict)
    final_diagnosis: dict = field(default_factory=dict)
    repair_attempts: list = field(default_factory=list)
    monitoring_events: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


def compile_diagnosis(session: DiagnosticSession) -> dict:
    """Compile full diagnosis from automated tool results (and legacy QA pairs if present)."""
    
    diagnostic_text = "AUTOMATED TOOL INSPECTIONS:\n"
    if session.automated_checks:
        for tool_name, result in session.automated_checks.items():
            diagnostic_text += f"\n--- {tool_name.upper()} ---\n{result}\n"
    else:
        diagnostic_text += "No automated checks run.\n"

    if session.qa_pairs:
        diagnostic_text += "\nLEGACY Q&A:\n"
        diagnostic_text += "\n\n".join([
            f"Q{i+1}: {qa['question']}\nA: {qa['answer']}\nAnalysis: {qa.get('analysis', 'N/A')}"
            for i, qa in enumerate(session.qa_pairs)
        ])

    messages = [
        {
            "role": "user",
            "content": f"""You are diagnosing an LLM problem. Here are the complete diagnostic findings pulled directly from the model files:

ORIGINAL PROBLEM:
{session.problem_description}

MODEL: {session.model_path}

INSPECTION RESULTS:
{diagnostic_text}

Analyze the hardware/weights/config status above.
CRITICAL INSTRUCTION: Your diagnosis MUST be grounded in the INSPECTION RESULTS. If the automated scans show that 'Inference Test Successful' and weights are healthy, you MUST state that the model is functioning properly on a hardware level, even if the user's ORIGINAL PROBLEM claims it "does not work" or "does not answer". Correct the user if the technical evidence contradicts their symptom.

Now provide a comprehensive diagnosis as JSON:
{{
  "primary_issue": "Main problem in 1 sentence based on actual file/config outputs",
  "affected_components": [
    {{"component": "name (e.g. config.json, weights file, generation_config)", "type": "layer/neuron/module/config", "severity": "critical/high/medium/low", "description": "what's wrong"}}
  ],
  "root_causes": ["cause1", "cause2"],
  "contributing_factors": ["factor1", "factor2"],
  "confidence": 0.0-1.0,
  "urgency": "immediate/high/medium/low",
  "summary": "2-3 paragraph technical summary synthesizing the automated file scan results"
}}"""
        }
    ]
    
    return get_json_completion(messages, system_prompt=DIAGNOSTICIAN_SYSTEM, max_tokens=3000)


def check_model_health(model_path: str, issue_description: str) -> dict:
    """Check specific model health by actually evaluating local files and returning real results."""
    
    # 1. Gather actual data
    print(f"  [Scan] Inspecting Model Files located at {model_path}...")
    files_res = run_tool("inspect_model_files", {"model_path": model_path})
    
    print(f"  [Scan] Checking Configuration...")
    config_res = run_tool("inspect_model_config", {"model_path": model_path})
    
    print(f"  [Scan] Validating Weights Integrity...")
    integrity_res = run_tool("validate_weights_integrity", {"model_path": model_path, "deep_check": False})
    
    print(f"  [Scan] Running Deep Neuron Layer Analysis...")
    neuron_res = run_tool("deep_neuron_inspection", {"model_path": model_path})
    
    print(f"  [Scan] Running Live Inference/Execution Test...")
    inference_res = run_tool("test_model_loading", {"model_path": model_path, "device": "cpu"})
    
    scan_details = f"File Inspection:\n{files_res.output}\n\n"
    scan_details += f"Config Inspection:\n{config_res.output}\n\n"
    scan_details += f"Weights Integrity:\n{integrity_res.output}\n\n"
    scan_details += f"Deep Neuron Analysis:\n{neuron_res.output}\n\n"
    scan_details += f"Live Execution/Inference:\n{inference_res.output}\n"
    
    # 2. Ask LLM to synthesize this hard data
    messages = [
        {
            "role": "user",
            "content": f"""Model: {model_path}
Reported Issue: {issue_description}

We have run actual hardware/file diagnostics on this model path. Here are the raw scan results:

{scan_details}

Produce a robust, conclusive Health Report payload summarizing these actual findings:
CRITICAL INSTRUCTION: Your verdict MUST be grounded in the scan results. If the scan shows 'Inference Test Successful', declare the model Healthy, even if the Reported Issue claims otherwise.

Return JSON:
{{
  "health_status": "Healthy/Degraded/Broken/Unknown",
  "what_we_checked": [
    {{"check_name": "Name of check performed", "result": "Pass/Fail", "details": "Specific finding from scan logs"}}
  ],
  "likely_problematic_areas": ["area based on REAL data above"],
  "healthy_indicators": ["good things found in scans"],
  "red_flags": ["critical warnings found in scans", "missing files", "empty weights"]
}}"""
        }
    ]
    return get_json_completion(messages, system_prompt=DIAGNOSTICIAN_SYSTEM)
