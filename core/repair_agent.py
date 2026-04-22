"""
Repair Agent - LLM-powered agent that uses tools to diagnose and fix model issues.
"""

import json
import re
from typing import Callable, Optional
from core.llm_client import get_completion, get_json_completion, stream_completion
from tools.repair_tools import run_tool, get_tool_list_for_llm, ToolResult
from core.diagnostics import DiagnosticSession


REPAIR_AGENT_SYSTEM = """You are an expert AI model repair engineer. You have access to 
specialized tools to inspect and fix language model configuration issues. 

CRITICAL RULES:
1. You NEVER modify model weights or training data - only configuration files
2. ALWAYS backup before making any changes (use backup_model_config first)
3. Be systematic: inspect first, then fix
4. After each fix, verify it worked
5. If a fix might break things, warn the user first
6. Only fix what is clearly wrong - don't change things that aren't broken

When you want to use a tool, respond with:
TOOL_CALL: {"tool": "tool_name", "params": {"param1": "value1"}}

When you're done, end with:
REPAIR_COMPLETE: {"fixed": [...], "remaining_issues": [...], "summary": "..."}"""


def plan_repairs(session: DiagnosticSession, stream_callback: Optional[Callable] = None) -> dict:
    """Have the LLM plan what repairs to make based on diagnosis."""
    diagnosis_str = json.dumps(session.final_diagnosis, indent=2)
    
    messages = [
        {
            "role": "user",
            "content": f"""Model: {session.model_path}
Problem: {session.problem_description}

Diagnosis:
{diagnosis_str}

Available tools:
{get_tool_list_for_llm()}

Create a repair plan. Return JSON:
{{
  "repair_steps": [
    {{
      "step": 1,
      "action": "what to do",
      "tool": "tool_name",
      "params": {{}},
      "expected_outcome": "what should happen",
      "risk": "low/medium/high"
    }}
  ],
  "estimated_fixes": "what this will resolve",
  "cannot_fix": "what requires retraining or manual intervention"
}}"""
        }
    ]
    
    return get_json_completion(messages, system_prompt=REPAIR_AGENT_SYSTEM)


def execute_repair_agent(
    session: DiagnosticSession,
    auto_fix: bool = False,
    stream_callback: Optional[Callable] = None,
    confirm_callback: Optional[Callable] = None,
) -> list[dict]:
    """
    Run the repair agent loop.
    
    Args:
        session: The diagnostic session with diagnosis
        auto_fix: If True, auto-apply fixes. If False, ask for confirmation
        stream_callback: Called with (text) for streaming output
        confirm_callback: Called with (action_description) returns bool
    
    Returns:
        List of repair attempt results
    """
    
    def emit(text):
        if stream_callback:
            stream_callback(text)
    
    repair_attempts = []
    
    # Build context for the agent
    diagnosis_str = json.dumps(session.final_diagnosis, indent=2)
    
    scan_summary = ""
    if session.automated_checks:
        for t, res in session.automated_checks.items():
            scan_summary += f"[{t.upper()}]\n{res}\n\n"
            
    qa_summary = ""
    if session.qa_pairs:
        qa_summary = "LEGACY Q&A:\n" + "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in session.qa_pairs])
    
    # Initial agent message
    agent_messages = [
        {
            "role": "user",
            "content": f"""You are repairing an LLM with the following issues:

MODEL PATH: {session.model_path}
ORIGINAL PROBLEM: {session.problem_description}

AUTOMATED SCAN RESULTS:
{scan_summary if scan_summary else "No automated scans found."}

{qa_summary}


DIAGNOSIS:
{diagnosis_str}

AVAILABLE TOOLS:
{get_tool_list_for_llm()}

Start by backing up the config, then systematically inspect and fix all issues you can.
Use TOOL_CALL format for each tool. After all repairs, end with REPAIR_COMPLETE."""
        }
    ]
    
    max_iterations = 20
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        emit(f"\n[Agent Step {iteration}]\n")
        
        # Get agent response
        response = get_completion(agent_messages, system_prompt=REPAIR_AGENT_SYSTEM, temperature=0.3, max_tokens=1500)
        emit(response + "\n")
        
        # Add to conversation
        agent_messages.append({"role": "assistant", "content": response})
        
        # Check for tool calls
        tool_call_match = re.search(r'TOOL_CALL:\s*(\{.*?\})', response, re.DOTALL)
        complete_match = re.search(r'REPAIR_COMPLETE:\s*(\{.*?\})', response, re.DOTALL)
        
        if complete_match:
            try:
                complete_data = json.loads(complete_match.group(1))
                emit(f"\n✅ Repair agent completed.\n")
                emit(f"Fixed: {complete_data.get('fixed', [])}\n")
                emit(f"Remaining: {complete_data.get('remaining_issues', [])}\n")
                repair_attempts.append({
                    "type": "completion",
                    "data": complete_data,
                    "success": True,
                })
            except:
                pass
            break
        
        if tool_call_match:
            try:
                tool_call = json.loads(tool_call_match.group(1))
                tool_name = tool_call.get("tool")
                tool_params = tool_call.get("params", {})
                
                emit(f"\n🔧 Calling tool: {tool_name}\n")
                emit(f"   Params: {json.dumps(tool_params, indent=2)}\n")
                
                # Confirm destructive operations if not auto_fix
                is_destructive = any(word in tool_name for word in ["fix", "restore", "add"])
                if is_destructive and not auto_fix and confirm_callback:
                    description = f"Apply: {tool_name}({tool_params})"
                    if not confirm_callback(description):
                        tool_result_text = "User declined this repair action."
                        emit(f"   ⏭️  Skipped by user\n")
                        agent_messages.append({"role": "user", "content": f"Tool result: {tool_result_text}"})
                        continue
                
                # Run the tool
                result = run_tool(tool_name, tool_params)
                
                attempt = {
                    "step": iteration,
                    "tool": tool_name,
                    "params": tool_params,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration,
                }
                repair_attempts.append(attempt)
                session.repair_attempts.append(attempt)
                
                if result.success:
                    emit(f"   ✅ Success:\n{result.output}\n")
                    tool_result_text = f"Tool succeeded:\n{result.output}"
                else:
                    emit(f"   ❌ Failed: {result.error}\n")
                    tool_result_text = f"Tool failed: {result.error}"
                
                agent_messages.append({"role": "user", "content": f"Tool result for {tool_name}:\n{tool_result_text}"})
                
            except json.JSONDecodeError as e:
                emit(f"   ⚠️  Could not parse tool call: {e}\n")
                agent_messages.append({"role": "user", "content": "Could not parse your TOOL_CALL JSON. Please check the format."})
        else:
            # No tool call and no completion - agent is thinking/explaining
            agent_messages.append({"role": "user", "content": "Continue with the next repair step, or use REPAIR_COMPLETE if done."})
        
        if iteration >= max_iterations:
            emit("\n⚠️  Max iterations reached. Stopping repair agent.\n")
    
    return repair_attempts


def get_repair_summary(repair_attempts: list, session: DiagnosticSession) -> dict:
    """Have the LLM summarize all repair actions taken."""
    attempts_text = json.dumps([
        {"tool": a.get("tool"), "success": a.get("success"), "output": a.get("output", "")[:200]}
        for a in repair_attempts if a.get("type") != "completion"
    ], indent=2)
    
    messages = [
        {
            "role": "user",
            "content": f"""Model: {session.model_path}
Original problem: {session.problem_description}

Repair actions taken:
{attempts_text}

Provide a clear repair summary as JSON:
{{
  "actions_taken": ["action1", "action2"],
  "issues_resolved": ["issue1", "issue2"],
  "issues_remaining": ["issue1"],
  "requires_retraining": true/false,
  "retraining_reason": "why if needed",
  "confidence_in_fix": 0.0-1.0,
  "next_steps": ["recommendation1"]
}}"""
        }
    ]
    
    return get_json_completion(messages, system_prompt=REPAIR_AGENT_SYSTEM)
