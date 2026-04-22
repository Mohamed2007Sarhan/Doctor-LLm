"""
Report Generator - Creates comprehensive diagnostic and repair reports.
"""

import json
import time
from datetime import datetime
from typing import Optional
from core.llm_client import get_completion
from core.diagnostics import DiagnosticSession


REPORT_SYSTEM = """You are an expert technical report writer specializing in AI/ML systems.
Write clear, precise, actionable reports. Use professional language.
Organize information logically with clear sections.
IMPORTANT: Use ONLY the real data provided. Do NOT create example content."""


def _ensure_diagnosis_complete(diagnosis: dict) -> dict:
    """Ensure diagnosis has all required fields."""
    defaults = {
        'primary_issue': 'Diagnosis completed',
        'affected_components': [],
        'root_causes': [],
        'contributing_factors': [],
        'confidence': 0.0,
        'urgency': 'medium',
        'summary': 'See analysis sections below',
        'next_steps': []
    }
    
    result = {**defaults}
    result.update({k: v for k, v in diagnosis.items() if v})  # Keep non-empty actual values
    return result


def generate_full_report(session: DiagnosticSession, monitor_status: Optional[dict] = None) -> str:
    """Generate a complete HTML diagnostic report."""
    
    diagnosis = session.final_diagnosis
    qa_pairs = session.qa_pairs
    repair_attempts = session.repair_attempts
    
    # Validate we have real data
    if not qa_pairs and not session.automated_checks and not session.identified_issues:
        raise ValueError("No diagnostic data found in session. Please complete the diagnostic process first.")
    
    if not diagnosis:
        raise ValueError("No diagnosis found in session. Please complete the diagnostic analysis first.")
    
    # Ensure diagnosis has all fields
    diagnosis = _ensure_diagnosis_complete(diagnosis)
    
    # Get LLM to write the narrative sections
    if qa_pairs:
        diag_evidence = "Questions Asked and Answered:\n" + "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
    elif session.automated_checks:
        diag_evidence = "Automated Diagnostic Checks:\n" + "\n".join([f"[{k}] {str(v)[:300]}..." for k, v in session.automated_checks.items()])
    else:
        diag_evidence = "No raw diagnostic evidence recorded."
    
    messages = [
        {
            "role": "user",
            "content": f"""Write a technical report narrative for this LLM diagnostic case.
Use ONLY the data provided below. Do NOT create example or placeholder content.

ACTUAL Problem Reported: {session.problem_description}
ACTUAL Model: {session.model_path}
ACTUAL Diagnostic Data: {json.dumps(diagnosis, indent=2)}

Diagnostic Evidence:
{diag_evidence}

ACTUAL Repair attempts: {len(repair_attempts)} actions taken
{f'Repairs: {json.dumps([{"tool": r.get("tool"), "success": r.get("success"), "summary": r.get("output", "")[0:100]} for r in repair_attempts if r.get("type") != "completion"], indent=2)}' if repair_attempts else 'No repairs attempted'}

Write 3 sections using ONLY the actual data provided above. Do NOT add examples or generic content:
1. EXECUTIVE_SUMMARY (2-3 sentences, summarizing the actual problem and diagnosis)
2. TECHNICAL_ANALYSIS (3-4 paragraphs, analyzing the diagnostic findings and evidence)
3. RECOMMENDATIONS (3-5 bullet points, based on actual identified issues)

Format as:
---EXECUTIVE_SUMMARY---
text here
---TECHNICAL_ANALYSIS---
text here
---RECOMMENDATIONS---
text here"""
        }
    ]
    
    narrative = get_completion(messages, system_prompt=REPORT_SYSTEM, temperature=0.4, max_tokens=2000)
    
    # Parse sections
    sections = {}
    current_section = None
    current_lines = []
    
    for line in narrative.split("\n"):
        if line.startswith("---") and line.endswith("---"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line.strip("-").strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()
    
    # Build affected components table
    components_html = ""
    for comp in diagnosis.get("affected_components", []):
        severity_colors = {
            "critical": "#ff4444",
            "high": "#ff8800", 
            "medium": "#ffcc00",
            "low": "#44bb44",
        }
        color = severity_colors.get(comp.get("severity", "medium"), "#888888")
        components_html += f"""
        <tr>
            <td class="component-name">{comp.get('component', 'Unknown')}</td>
            <td><span class="type-badge">{comp.get('type', 'N/A')}</span></td>
            <td><span class="severity-badge" style="background:{color}">{comp.get('severity', 'N/A').upper()}</span></td>
            <td>{comp.get('description', 'N/A')}</td>
        </tr>"""
    
    # Build repair log
    repair_log_html = ""
    for i, attempt in enumerate(repair_attempts, 1):
        if attempt.get("type") == "completion":
            continue
        status_icon = "✅" if attempt.get("success") else "❌"
        repair_log_html += f"""
        <div class="repair-entry {'success' if attempt.get('success') else 'failure'}">
            <div class="repair-header">
                <span class="repair-step">Step {i}</span>
                <span class="repair-tool">🔧 {attempt.get('tool', 'unknown')}</span>
                <span class="repair-status">{status_icon}</span>
                <span class="repair-time">{attempt.get('duration', 0):.2f}s</span>
            </div>
            <div class="repair-output">{attempt.get('output', attempt.get('error', 'No output'))[:400]}</div>
        </div>"""
    
    # Q&A section
    qa_html = ""
    for i, qa in enumerate(qa_pairs, 1):
        qa_html += f"""
        <div class="qa-item">
            <div class="qa-question"><span class="qa-number">Q{i}</span> {qa.get('question', '')}</div>
            <div class="qa-answer">{qa.get('answer', '')}</div>
            {f'<div class="qa-analysis"><em>💡 {qa.get("analysis", "")}</em></div>' if qa.get('analysis') else ''}
        </div>"""
    
    # Monitor section
    monitor_html = ""
    if monitor_status:
        alerts_html = ""
        for alert in monitor_status.get("recent_alerts", []):
            color = {"critical": "#ff4444", "warning": "#ff8800", "info": "#4488ff"}.get(alert.get("severity", "info"), "#888")
            alerts_html += f'<div class="monitor-alert" style="border-left:4px solid {color}">{alert.get("severity", "").upper()}: {alert.get("issue", "")}</div>'
        
        monitor_html = f"""
        <section class="report-section monitoring">
            <h2>📡 Monitoring Status</h2>
            <div class="monitor-grid">
                <div class="monitor-stat"><span class="stat-label">Status</span><span class="stat-value {'running' if monitor_status.get('is_running') else 'stopped'}">{'🟢 Running' if monitor_status.get('is_running') else '🔴 Stopped'}</span></div>
                <div class="monitor-stat"><span class="stat-label">Snapshots</span><span class="stat-value">{monitor_status.get('snapshots_recorded', 0)}</span></div>
                <div class="monitor-stat"><span class="stat-label">Total Alerts</span><span class="stat-value">{monitor_status.get('total_alerts', 0)}</span></div>
                <div class="monitor-stat"><span class="stat-label">Critical</span><span class="stat-value critical">{monitor_status.get('critical_alerts', 0)}</span></div>
                <div class="monitor-stat"><span class="stat-label">Latest Loss</span><span class="stat-value">{monitor_status.get('latest_loss', 'N/A')}</span></div>
                <div class="monitor-stat"><span class="stat-label">Latest Step</span><span class="stat-value">{monitor_status.get('latest_step', 'N/A')}</span></div>
            </div>
            {f'<div class="recent-alerts">{alerts_html}</div>' if alerts_html else ''}
        </section>"""
    
    # Overall health score
    confidence = diagnosis.get("confidence", 0.5)
    urgency = diagnosis.get("urgency", "medium")
    fixed_count = sum(1 for a in repair_attempts if a.get("success") and a.get("type") != "completion")
    
    timestamp = datetime.fromtimestamp(session.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare recommendations for rendering (avoid backslash in f-string expression)
    recommendations_list = []
    if sections.get('RECOMMENDATIONS'):
        recommendations_list = [r.strip() for r in sections.get('RECOMMENDATIONS', '').split('\n') if r.strip()]
    elif diagnosis.get('next_steps'):
        recommendations_list = diagnosis.get('next_steps', [])
    
    # If still empty, build from root causes and contributing factors
    if not recommendations_list:
        recommendations_list = [f"Address {cause}" for cause in diagnosis.get('root_causes', [])]
        if not recommendations_list and diagnosis.get('contributing_factors'):
            recommendations_list = [f"Mitigate {factor}" for factor in diagnosis.get('contributing_factors', [])]
    
    # Fallback only if truly empty
    if not recommendations_list:
        recommendations_list = ["Review diagnosis above for actionable insights"]
    
    recommendations_html = ''.join(
        f'<div class="rec-item"><div class="rec-num">{i+1}</div><div class="rec-text">{rec}</div></div>'
        for i, rec in enumerate(recommendations_list)
    )
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Doctor Report — {session.model_path}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
  
  :root {{
    --bg: #0a0c12;
    --surface: #111520;
    --surface2: #161b2a;
    --border: #1e2740;
    --accent: #00d4ff;
    --accent2: #7b61ff;
    --accent3: #ff6b35;
    --text: #e2e8f8;
    --text-dim: #8892aa;
    --critical: #ff4444;
    --warning: #ff8800;
    --success: #22dd88;
    --info: #4488ff;
  }}
  
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }}
  
  /* Background grid */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: 
      linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }}
  
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; position: relative; z-index: 1; }}
  
  /* Header */
  .report-header {{
    border: 1px solid var(--border);
    background: var(--surface);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }}
  
  .report-header::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  }}
  
  .header-top {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 24px;
  }}
  
  .logo {{
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    opacity: 0.8;
  }}
  
  .report-title {{
    font-size: 36px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 8px;
  }}
  
  .report-title span {{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  
  .report-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-dim);
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
  }}
  
  .model-path {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--accent);
    margin-bottom: 20px;
    word-break: break-all;
  }}
  
  /* Status badges */
  .status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }}
  
  .status-badge {{
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid;
  }}
  
  .urgency-immediate {{ background: rgba(255,68,68,0.15); color: #ff4444; border-color: #ff4444; }}
  .urgency-high {{ background: rgba(255,136,0,0.15); color: #ff8800; border-color: #ff8800; }}
  .urgency-medium {{ background: rgba(255,204,0,0.15); color: #ffcc00; border-color: #ffcc00; }}
  .urgency-low {{ background: rgba(68,187,68,0.15); color: #44bb44; border-color: #44bb44; }}
  
  /* Confidence meter */
  .confidence-meter {{
    flex: 1;
    min-width: 200px;
  }}
  
  .confidence-label {{
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
  }}
  
  .confidence-bar {{
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}
  
  .confidence-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 3px;
    width: {confidence * 100:.0f}%;
    transition: width 1s ease;
  }}
  
  /* Sections */
  .report-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
  }}
  
  .report-section h2 {{
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  
  /* Executive summary */
  .executive-text {{
    font-size: 16px;
    line-height: 1.7;
    color: var(--text);
    padding: 20px;
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,97,255,0.05));
    border-radius: 12px;
    border-left: 3px solid var(--accent);
  }}
  
  /* Affected components table */
  .components-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}
  
  .components-table th {{
    text-align: left;
    padding: 12px 16px;
    background: var(--surface2);
    color: var(--text-dim);
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }}
  
  .components-table td {{
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  
  .components-table tr:last-child td {{ border-bottom: none; }}
  .component-name {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent); }}
  
  .severity-badge {{
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
    color: white;
  }}
  
  .type-badge {{
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent2);
  }}
  
  /* Root causes */
  .causes-list, .factors-list {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  
  .causes-list li, .factors-list li {{
    padding: 12px 16px;
    background: var(--surface2);
    border-radius: 8px;
    border-left: 3px solid var(--accent3);
    font-size: 14px;
    line-height: 1.5;
  }}
  
  .factors-list li {{ border-left-color: var(--accent2); }}
  
  /* Technical analysis */
  .tech-analysis {{
    font-size: 14px;
    line-height: 1.8;
    color: var(--text);
    white-space: pre-wrap;
  }}
  
  /* Q&A */
  .qa-item {{
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }}
  
  .qa-item:last-child {{ border-bottom: none; }}
  
  .qa-question {{
    font-size: 14px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }}
  
  .qa-number {{
    background: var(--accent);
    color: var(--bg);
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 12px;
    white-space: nowrap;
    margin-top: 1px;
  }}
  
  .qa-answer {{
    font-size: 13px;
    line-height: 1.6;
    color: var(--text);
    padding-left: 38px;
  }}
  
  .qa-analysis {{
    font-size: 12px;
    color: var(--text-dim);
    padding-left: 38px;
    margin-top: 8px;
  }}
  
  /* Repair log */
  .repair-entry {{
    background: var(--surface2);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 3px solid var(--border);
  }}
  
  .repair-entry.success {{ border-left-color: var(--success); }}
  .repair-entry.failure {{ border-left-color: var(--critical); }}
  
  .repair-header {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }}
  
  .repair-step {{ font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }}
  .repair-tool {{ font-size: 13px; font-weight: 700; color: var(--text); }}
  .repair-status {{ font-size: 16px; }}
  .repair-time {{ font-size: 11px; color: var(--text-dim); margin-left: auto; font-family: 'JetBrains Mono', monospace; }}
  
  .repair-output {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    white-space: pre-wrap;
    background: rgba(0,0,0,0.3);
    padding: 10px;
    border-radius: 6px;
    max-height: 120px;
    overflow-y: auto;
  }}
  
  /* Monitoring */
  .monitor-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }}
  
  .monitor-stat {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  
  .stat-label {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em; }}
  .stat-value {{ font-size: 18px; font-weight: 700; color: var(--text); font-family: 'JetBrains Mono', monospace; }}
  .stat-value.running {{ color: var(--success); }}
  .stat-value.critical {{ color: var(--critical); }}
  
  .monitor-alert {{
    padding: 10px 16px;
    background: var(--surface2);
    border-radius: 6px;
    font-size: 13px;
    margin-bottom: 8px;
    color: var(--text);
  }}
  
  /* Recommendations */
  .recommendations-list {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  
  .rec-item {{
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 14px 16px;
    background: var(--surface2);
    border-radius: 8px;
  }}
  
  .rec-num {{
    width: 28px; height: 28px;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    color: white;
  }}
  
  .rec-text {{ font-size: 14px; line-height: 1.6; }}
  
  /* Footer */
  .report-footer {{
    text-align: center;
    padding: 24px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }}
  
  /* Summary stats bar */
  .stats-bar {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1px;
    background: var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 24px;
  }}
  
  .stat-block {{
    background: var(--surface2);
    padding: 16px;
    text-align: center;
  }}
  
  .stat-block .num {{ font-size: 28px; font-weight: 800; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; }}
  .stat-block .lbl {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="report-header">
    <div class="header-top">
      <div>
        <div class="logo">⚕️ LLM Doctor — Diagnostic Report</div>
        <div class="report-title">Model <span>Health Analysis</span></div>
      </div>
      <div class="report-meta">
        <span>Generated: {timestamp}</span>
        <span>Session ID: {int(session.timestamp)}</span>
      </div>
    </div>
    
    <div class="model-path">📁 {session.model_path or "No model path specified"}</div>
    
    <div class="status-row">
      <span class="status-badge urgency-{urgency}">
        ⚡ Urgency: {urgency.upper()}
      </span>
      <span class="status-badge" style="background:rgba(68,187,68,0.15);color:#44bb44;border-color:#44bb44">
        🔧 {fixed_count} Fixes Applied
      </span>
      <span class="status-badge" style="background:rgba(68,136,255,0.15);color:#4488ff;border-color:#4488ff">
        📊 {len(qa_pairs) if qa_pairs else len(session.automated_checks)} { "Questions Asked" if qa_pairs else "Checks Run" }
      </span>
      <div class="confidence-meter">
        <div class="confidence-label">
          <span>Diagnosis Confidence</span>
          <span>{confidence*100:.0f}%</span>
        </div>
        <div class="confidence-bar"><div class="confidence-fill"></div></div>
      </div>
    </div>
  </div>

  <!-- Stats Bar -->
  <div class="stats-bar">
    <div class="stat-block"><span class="num">{len(qa_pairs) if qa_pairs else len(session.automated_checks)}</span><span class="lbl">{"Questions" if qa_pairs else "Checks"}</span></div>
    <div class="stat-block"><span class="num">{len(diagnosis.get('affected_components', []))}</span><span class="lbl">Components</span></div>
    <div class="stat-block"><span class="num">{len(diagnosis.get('root_causes', []))}</span><span class="lbl">Root Causes</span></div>
    <div class="stat-block"><span class="num">{len(repair_attempts)}</span><span class="lbl">Repair Steps</span></div>
    <div class="stat-block"><span class="num">{fixed_count}</span><span class="lbl">Fixed</span></div>
    <div class="stat-block"><span class="num">{confidence*100:.0f}%</span><span class="lbl">Confidence</span></div>
  </div>

  <!-- Executive Summary -->
  <section class="report-section">
    <h2>📋 Executive Summary</h2>
    <div class="executive-text">
      {sections.get('EXECUTIVE_SUMMARY') or f"Session analyzed {len(qa_pairs) if qa_pairs else len(session.automated_checks)} data points. Primary issue: {diagnosis.get('primary_issue', 'See analysis below for details')}."}
    </div>
  </section>

  <!-- Primary Issue -->
  <section class="report-section">
    <h2>🎯 Primary Diagnosis</h2>
    <div class="executive-text" style="border-left-color: var(--accent3); font-size:15px;">
      {diagnosis.get('primary_issue') or 'Analysis complete - see affected components and root causes below'}
    </div>
  </section>

  <!-- Affected Components -->
  {'<section class="report-section"><h2>🧠 Affected Components</h2><table class="components-table"><thead><tr><th>Component</th><th>Type</th><th>Severity</th><th>Description</th></tr></thead><tbody>' + components_html + '</tbody></table></section>' if components_html else ''}

  <!-- Root Causes -->
  {'<section class="report-section"><h2>🔍 Root Causes</h2><ul class="causes-list">' + ''.join(f'<li>{c}</li>' for c in diagnosis.get('root_causes', [])) + '</ul></section>' if diagnosis.get('root_causes') else ''}

  <!-- Contributing Factors -->
  {'<section class="report-section"><h2>⚠️ Contributing Factors</h2><ul class="factors-list">' + ''.join(f'<li>{f}</li>' for f in diagnosis.get('contributing_factors', [])) + '</ul></section>' if diagnosis.get('contributing_factors') else ''}

  <!-- Technical Analysis -->
  <section class="report-section">
    <h2>🔬 Technical Analysis</h2>
    <div class="tech-analysis">{sections.get('TECHNICAL_ANALYSIS') or diagnosis.get('summary') or 'Technical analysis based on diagnostic interview and findings'}</div>
  </section>

  <!-- Diagnostic Evidence -->
  <section class="report-section">
    <h2>💬 Diagnostic Evidence</h2>
    <div>{qa_html if qa_html else ('<div class="qa-item"><div class="qa-question">Automated Checks</div><div class="qa-answer">' + "<br>".join([f"<strong>{k}:</strong> {str(v)[:200]}..." for k, v in session.automated_checks.items()]) + '</div></div>' if session.automated_checks else '<p style="color:var(--text-dim)">No diagnostic evidence recorded</p>')}</div>
  </section>

  <!-- Repair Log -->
  {'<section class="report-section"><h2>🔧 Repair Actions Log</h2>' + repair_log_html + '</section>' if repair_log_html else ''}

  <!-- Monitoring -->
  {monitor_html if monitor_html else ''}

  <!-- Recommendations -->
  <section class="report-section">
    <h2>💡 Recommendations</h2>
    <div class="recommendations-list">
      {recommendations_html}
    </div>
  </section>

  <div class="report-footer">
    ⚕️ LLM Doctor &nbsp;|&nbsp; Powered by NVIDIA Llama-3.3-Nemotron-Super-49B &nbsp;|&nbsp; {timestamp} &nbsp;|&nbsp; Session {int(session.timestamp)}
  </div>

</div>
</body>
</html>"""
    
    return html
