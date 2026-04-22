"""
LLM Doctor — Main Application
A complete LLM diagnostic, repair, and monitoring tool.
"""

import os
import sys
import json
import time
import threading
import webbrowser
import tempfile
from pathlib import Path

# Rich terminal UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install rich for better UI: pip install rich")

# Transformers
try:
    from transformers import AutoConfig, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.diagnostics import DiagnosticSession, compile_diagnosis, check_model_health
from core.repair_agent import execute_repair_agent, plan_repairs, get_repair_summary
from core.monitor import ModelMonitor, MetricSnapshot
from reports.report_generator import generate_full_report
from core.llm_client import MODEL_INFO


console = Console() if RICH_AVAILABLE else None

# ─── BANNER ─────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██╗     ██╗     ███╗   ███╗    ██████╗  ██████╗  ██████╗ ████████╗      ║
║    ██║     ██║     ████╗ ████║    ██╔══██╗██╔═══██╗██╔════╝ ╚══██╔══╝      ║
║    ██║     ██║     ██╔████╔██║    ██║  ██║██║   ██║██║         ██║          ║
║    ██║     ██║     ██║╚██╔╝██║    ██║  ██║██║   ██║██║         ██║          ║
║    ███████╗███████╗██║ ╚═╝ ██║    ██████╔╝╚██████╔╝╚██████╗    ██║          ║
║    ╚══════╝╚══════╝╚═╝     ╚═╝    ╚═════╝  ╚═════╝  ╚═════╝    ╚═╝          ║
║                                                                              ║
║                    ⚕️  AI Model Diagnostics & Repair System                  ║
║                    Powered by NVIDIA Llama-3.3-Nemotron-Super-49B            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    if RICH_AVAILABLE:
        console.print(BANNER, style="bold cyan")
    else:
        print(BANNER)


def print_section(title: str, emoji: str = ""):
    if RICH_AVAILABLE:
        console.print(Rule(f"[bold cyan]{emoji} {title}[/bold cyan]", style="cyan"))
    else:
        print(f"\n{'='*70}")
        print(f"  {emoji} {title}")
        print(f"{'='*70}")


def print_info(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[dim]  {msg}[/dim]")
    else:
        print(f"  {msg}")


def print_success(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold green]  ✅ {msg}[/bold green]")
    else:
        print(f"  ✅ {msg}")


def print_warning(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]  ⚠️  {msg}[/bold yellow]")
    else:
        print(f"  ⚠️  {msg}")


def print_error(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold red]  ❌ {msg}[/bold red]")
    else:
        print(f"  ❌ {msg}")


def get_input(prompt: str) -> str:
    if RICH_AVAILABLE:
        val = Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]")
    else:
        val = input(f"\n{prompt}: ")
        
    val = val.strip()
    # Remove surrounding quotes (common when copying paths in Windows)
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1]
    elif val.startswith("'") and val.endswith("'"):
        val = val[1:-1]
        
    return val


def get_confirm(prompt: str, default: bool = False) -> bool:
    if RICH_AVAILABLE:
        return Confirm.ask(f"[bold yellow]{prompt}[/bold yellow]", default=default)
    else:
        resp = input(f"\n{prompt} [y/n]: ").strip().lower()
        return resp in ("y", "yes")


# ─── PHASE 1: INTAKE ────────────────────────────────────────────────────────

def intake_phase() -> DiagnosticSession:
    """Collect initial problem description and model path."""
    print_section("Patient Intake", "📋")
    
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold white]Welcome to LLM Doctor![/bold white]\n\n"
            "I'm going to ask you about your model and its problems.\n"
            "Please be as [bold cyan]detailed as possible[/bold cyan] — the more you tell me, the better I can diagnose.\n\n"
            "[dim]This session will cover:[/dim]\n"
            "[dim]  1. Problem description & model path[/dim]\n"
            "[dim]  2. 10 diagnostic questions[/dim]\n"
            "[dim]  3. Automated diagnosis[/dim]\n"
            "[dim]  4. Repair options[/dim]\n"
            "[dim]  5. Full report generation[/dim]",
            title="[bold cyan]⚕️ LLM Doctor[/bold cyan]",
            border_style="cyan",
        ))
    
    print_section("Step 1: Describe Your Problem", "🔍")
    print_info("Describe everything you're experiencing with your model.")
    print_info("Include: symptoms, when it started, what you've tried, error messages, etc.")
    print_info("")
    
    if RICH_AVAILABLE:
        console.print("[bold white]Problem Description[/bold white] (press Enter when done):")
        problem = input().strip()
    else:
        print("\nDescribe the problem (press Enter when done):")
        problem = input().strip()
    
    if not problem:
        problem = "Model is not performing as expected"
    
    print_section("Step 2: Model Path", "📁")
    print_info("Enter the path or name of your model (e.g., /path/to/model or meta-llama/Llama-2-7b)")
    
    model_path = get_input("Model path or name")
    if not model_path:
        model_path = "unknown_model"
    
    session = DiagnosticSession(
        problem_description=problem,
        model_path=model_path,
    )
    
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold white]Problem:[/bold white] {problem[:200]}{'...' if len(problem) > 200 else ''}\n\n"
            f"[bold white]Model:[/bold white] [cyan]{model_path}[/cyan]",
            title="[green]✅ Intake Complete[/green]",
            border_style="green",
        ))
    
    return session


# ─── PHASE 2: AUTOMATED DIAGNOSTICS ──────────────────────────────────────────

def automated_diagnostic_phase(session: DiagnosticSession) -> DiagnosticSession:
    """Run automated file and config inspections."""
    print_section("Automated Diagnostics", "🩺")
    
    print_info("Analyzing model files, configurations, and weights validity...")
    print_info("")
    
    from tools.repair_tools import run_tool
    
    checks_to_run = [
        {"name": "inspect_model_files", "params": {"model_path": session.model_path}, "desc": "Inspecting files"},
        {"name": "inspect_model_config", "params": {"model_path": session.model_path}, "desc": "Checking configuration"},
        {"name": "check_tokenizer_config", "params": {"model_path": session.model_path}, "desc": "Checking tokenizer"},
        {"name": "validate_weights_integrity", "params": {"model_path": session.model_path, "deep_check": False}, "desc": "Validating weights integrity"},
        {"name": "deep_neuron_inspection", "params": {"model_path": session.model_path}, "desc": "Running deep neuron and layer analysis"},
        {"name": "test_model_loading", "params": {"model_path": session.model_path, "device": "cpu"}, "desc": "Running live inference test (Monitoring execution)"},
    ]
    
    if not hasattr(session, "automated_checks"):
        session.automated_checks = {}
    
    if RICH_AVAILABLE:
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console) as progress:
            for check in checks_to_run:
                task = progress.add_task(f"{check['desc']}...", total=None)
                res = run_tool(check["name"], check["params"])
                session.automated_checks[check["name"]] = res.output
                progress.stop_task(task)
                progress.update(task, description=f"[green]✓ {check['desc']}[/green]")
    else:
        for check in checks_to_run:
            print_info(f"Running: {check['desc']}...")
            res = run_tool(check["name"], check["params"])
            session.automated_checks[check["name"]] = res.output
            if res.success:
                print_success(f"{check['desc']} complete.")
            else:
                print_warning(f"{check['desc']} reported issues.")
                
    print_success("Automated diagnostics complete!")
    return session


# ─── PHASE 3: COMPILE DIAGNOSIS ─────────────────────────────────────────────

def compile_diagnosis_phase(session: DiagnosticSession) -> DiagnosticSession:
    """Compile the full diagnosis from all Q&A data."""
    print_section("Compiling Diagnosis", "🔬")
    print_info("Analyzing all answers and building comprehensive diagnosis...")
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running full diagnostic analysis...", total=None)
            diagnosis = compile_diagnosis(session)
            progress.stop()
    else:
        diagnosis = compile_diagnosis(session)
    
    session.final_diagnosis = diagnosis
    
    # Display diagnosis summary
    if RICH_AVAILABLE:
        # Main finding
        console.print(Panel(
            f"[bold white]{diagnosis.get('primary_issue', 'See components below')}[/bold white]",
            title="[red]🎯 Primary Issue[/red]",
            border_style="red",
        ))
        
        # Affected components table
        if diagnosis.get("affected_components"):
            table = Table(title="Affected Components", border_style="cyan")
            table.add_column("Component", style="cyan bold")
            table.add_column("Type", style="dim")
            table.add_column("Severity", style="bold")
            table.add_column("Description")
            
            severity_styles = {
                "critical": "bold red",
                "high": "bold orange3",
                "medium": "bold yellow",
                "low": "bold green",
            }
            
            for comp in diagnosis.get("affected_components", []):
                sev = comp.get("severity", "medium")
                table.add_row(
                    comp.get("component", "?"),
                    comp.get("type", "?"),
                    Text(sev.upper(), style=severity_styles.get(sev, "white")),
                    comp.get("description", "")[:80],
                )
            console.print(table)
        
        # Root causes
        if diagnosis.get("root_causes"):
            console.print("\n[bold red]Root Causes:[/bold red]")
            for rc in diagnosis["root_causes"]:
                console.print(f"  [red]•[/red] {rc}")
        
        urgency = diagnosis.get("urgency", "medium")
        confidence = diagnosis.get("confidence", 0.5)
        console.print(f"\n  [bold]Urgency:[/bold] [yellow]{urgency.upper()}[/yellow]  |  [bold]Confidence:[/bold] [cyan]{confidence*100:.0f}%[/cyan]")
    else:
        print(f"\nPrimary Issue: {diagnosis.get('primary_issue', 'N/A')}")
        print(f"Urgency: {diagnosis.get('urgency', 'N/A')}")
        print(f"Confidence: {diagnosis.get('confidence', 0)*100:.0f}%")
    
    return session


# ─── PHASE 4: REPAIR ────────────────────────────────────────────────────────

def repair_phase(session: DiagnosticSession) -> DiagnosticSession:
    """Optionally run the repair agent."""
    print_section("Repair Options", "🔧")
    
    do_repair = get_confirm("Would you like to attempt automated repair of identified issues?", default=True)
    
    if not do_repair:
        print_info("Skipping repair phase. Report will include recommendations.")
        return session
    
    # Check if model path exists for actual file operations
    model_path_exists = Path(session.model_path).exists()
    
    if not model_path_exists:
        print_warning(f"Model path '{session.model_path}' does not exist locally.")
        print_info("The repair agent will still plan repairs and provide guidance.")
        print_info("For actual file fixes, ensure the model path is accessible.")
    
    auto_fix = get_confirm(
        "Enable AUTO-FIX mode? (If No, you'll confirm each repair action individually)",
        default=False
    )
    
    if auto_fix:
        print_warning("AUTO-FIX mode enabled. The agent will apply fixes without asking for each one.")
    else:
        print_info("Manual mode: You'll be asked to confirm each repair action.")
    
    print_info("")
    print_info("Starting repair agent...")
    print_info("")
    
    repair_log = []
    
    def stream_output(text: str):
        if RICH_AVAILABLE:
            console.print(text, end="")
        else:
            print(text, end="", flush=True)
    
    def confirm_repair(description: str) -> bool:
        return get_confirm(f"Apply repair: {description[:100]}?", default=True)
    
    repair_log = execute_repair_agent(
        session,
        auto_fix=auto_fix,
        stream_callback=stream_output,
        confirm_callback=confirm_repair if not auto_fix else None,
    )
    
    # Get repair summary
    print_info("\nGenerating repair summary...")
    repair_summary = get_repair_summary(repair_log, session)
    
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold white]Fixed:[/bold white] {repair_summary.get('issues_resolved', [])}\n\n"
            f"[bold white]Remaining:[/bold white] {repair_summary.get('issues_remaining', [])}\n\n"
            f"[bold white]Requires Retraining:[/bold white] {'Yes — ' + repair_summary.get('retraining_reason', '') if repair_summary.get('requires_retraining') else 'No'}\n\n"
            f"[bold white]Confidence in Fix:[/bold white] {repair_summary.get('confidence_in_fix', 0)*100:.0f}%",
            title="[green]🔧 Repair Summary[/green]",
            border_style="green",
        ))
    else:
        print("\nRepair Summary:")
        print(json.dumps(repair_summary, indent=2))
    
    return session


# ─── PHASE 5: MONITORING ────────────────────────────────────────────────────

def monitoring_phase() -> tuple[bool, ModelMonitor | None]:
    """Optionally start continuous monitoring."""
    print_section("Continuous Monitoring", "📡")
    
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold white]Monitoring Mode[/bold white]\n\n"
            "Enable continuous monitoring to watch your model during training or inference.\n\n"
            "The monitor will:\n"
            "  📊 Track loss, gradient norms, memory usage, throughput\n"
            "  ⚠️  Alert you immediately when issues are detected\n"
            "  🔧 Auto-fix certain issues if AUTO-FIX is enabled\n"
            "  📝 Log all events for the final report",
            title="[cyan]📡 Monitoring Options[/cyan]",
            border_style="cyan",
        ))
    
    do_monitor = get_confirm("Enable continuous monitoring?", default=False)
    
    if not do_monitor:
        return False, None
    
    model_path = get_input("Model path to monitor")
    auto_fix = get_confirm("Enable AUTO-FIX for monitoring alerts?", default=False)
    log_file = get_input("Training log file to watch (leave empty to skip file monitoring)")
    
    alert_count = [0]
    
    def on_alert(alert):
        alert_count[0] += 1
        severity_colors = {"critical": "red", "warning": "yellow", "info": "blue"}
        color = severity_colors.get(alert.severity, "white")
        if RICH_AVAILABLE:
            console.print(f"\n[bold {color}]  🚨 ALERT [{alert.severity.upper()}]: {alert.issue}[/bold {color}]")
            console.print(f"  [dim]Action: {alert.suggested_action}[/dim]")
        else:
            print(f"\n🚨 ALERT [{alert.severity.upper()}]: {alert.issue}")
    
    monitor = ModelMonitor(
        model_path=model_path,
        auto_fix=auto_fix,
        alert_callback=on_alert,
        check_interval_seconds=5.0,
    )
    
    if log_file and Path(log_file).exists():
        monitor.start_file_monitoring(log_file)
        print_success(f"Monitoring log file: {log_file}")
    else:
        monitor.is_running = True
        print_info("Monitor initialized (no log file). Push metrics manually via monitor.push_metrics()")
    
    print_success("Monitoring active! Alerts will appear in real-time.")
    print_info("The monitor will run in the background during repair and report phases.")
    
    return True, monitor


# ─── PHASE 6: REPORT ─────────────────────────────────────────────────────────

def report_phase(session: DiagnosticSession, monitor: ModelMonitor = None):
    """Generate and display the final report."""
    print_section("Generating Report", "📊")
    
    # Validate session has real data
    if not getattr(session, 'automated_checks', {}) and not session.qa_pairs:
        print_error("Cannot generate report: No diagnostic data found.")
        print_error("Please complete the diagnostic phase first.")
        return
    
    if not session.final_diagnosis:
        print_error("Cannot generate report: No diagnosis compiled.")
        print_error("Please complete the diagnostic analysis first.")
        return
    
    print_info("Building comprehensive diagnostic report with your actual diagnostic data...")
    
    monitor_status = monitor.get_status() if monitor else None
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating HTML report from session data...", total=None)
            try:
                html_report = generate_full_report(session, monitor_status)
            except ValueError as e:
                progress.stop()
                print_error(f"Report generation failed: {e}")
                return
            progress.stop()
    else:
        try:
            html_report = generate_full_report(session, monitor_status)
        except ValueError as e:
            print_error(f"Report generation failed: {e}")
            return
    
    # Save report
    timestamp = int(session.timestamp)
    report_filename = f"llm_doctor_report_{timestamp}.html"
    report_path = Path.cwd() / report_filename
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    
    # Save session JSON
    session_filename = f"llm_doctor_session_{timestamp}.json"
    session_path = Path.cwd() / session_filename
    with open(session_path, "w") as f:
        f.write(session.to_json())
    
    print_success(f"HTML Report saved: {report_path}")
    print_success(f"Session data saved: {session_path}")
    
    # Open in browser
    open_browser = get_confirm("Open report in browser?", default=True)
    if open_browser:
        webbrowser.open(f"file://{report_path.absolute()}")
        print_success("Report opened in browser!")
    
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold white]Report:[/bold white] [cyan]{report_path}[/cyan]\n"
            f"[bold white]Session:[/bold white] [cyan]{session_path}[/cyan]\n\n"
            f"[dim]The report includes:[/dim]\n"
            f"[dim]  ✅ Full diagnosis with affected components[/dim]\n"
            f"[dim]  ✅ Root cause analysis[/dim]\n"
            f"[dim]  ✅ Complete diagnostic interview[/dim]\n"
            f"[dim]  ✅ Repair actions log[/dim]\n"
            f"[dim]  ✅ Monitoring status[/dim]\n"
            f"[dim]  ✅ Recommendations[/dim]",
            title="[bold green]✅ Report Complete[/bold green]",
            border_style="green",
        ))


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print_banner()
    
    if RICH_AVAILABLE:
        console.print(f"[dim]  Model: {MODEL_INFO['name']}[/dim]")
        console.print(f"[dim]  Provider: {MODEL_INFO['provider']}[/dim]")
        console.print(f"[dim]  {MODEL_INFO['description']}[/dim]\n")
    
    # ── Main menu ────────────────────────────────────────────────────────────
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold]Choose a mode:[/bold]\n\n"
            "  [cyan]1[/cyan]. 🩺 Full Diagnostic + Repair (recommended)\n"
            "  [cyan]2[/cyan]. 📡 Monitoring Only\n"
            "  [cyan]3[/cyan]. 📊 Load Previous Session & Repair\n"
            "  [cyan]4[/cyan]. ℹ️  Model Info Check",
            title="[bold cyan]LLM Doctor — Main Menu[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print("\n  Modes:\n  1. Full Diagnostic + Repair\n  2. Monitoring Only\n  3. Load Previous Session\n  4. Model Info Check")
    
    mode = get_input("Select mode [1/2/3/4]").strip()
    
    if mode == "2":
        _, monitor = monitoring_phase()
        if monitor:
            print_info("Monitoring running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(30)
                    status = monitor.get_status()
                    print_info(f"Status: {status['snapshots_recorded']} snapshots | {status['total_alerts']} alerts")
            except KeyboardInterrupt:
                monitor.stop()
                print_success("Monitoring stopped.")
        return
    
    elif mode == "3":
        session_file = get_input("Session JSON file path")
        try:
            with open(session_file) as f:
                data = json.load(f)
            session = DiagnosticSession(**{k: v for k, v in data.items() if k in DiagnosticSession.__dataclass_fields__})
            print_success("Session loaded!")
            session = repair_phase(session)
            report_phase(session)
        except Exception as e:
            print_error(f"Failed to load session: {e}")
        return
    
    elif mode == "4":
        model_path = get_input("Model path to inspect")
        issue = get_input("Describe any known issues (or press Enter for general check)")
        
        from core.diagnostics import check_model_health
        result = check_model_health(model_path, issue or "general health check")
        
        if RICH_AVAILABLE:
            console.print(Panel(json.dumps(result, indent=2), title="Model Health Check", border_style="cyan"))
        else:
            print(json.dumps(result, indent=2))
        return
    
    # Mode 1: Full diagnostic
    # ── Phase 1: Intake ──────────────────────────────────────────────────────
    session = intake_phase()
    
    # ── Phase 2: Monitoring setup (optional, start early) ────────────────────
    monitoring_active, monitor = monitoring_phase()
    
    # ── Phase 3: Automated diagnostics ───────────────────────────────────────
    session = automated_diagnostic_phase(session)
    
    # ── Phase 4: Compile diagnosis ────────────────────────────────────────────
    session = compile_diagnosis_phase(session)
    
    # ── Phase 5: Repair ───────────────────────────────────────────────────────
    session = repair_phase(session)
    
    # ── Phase 6: Final monitoring analysis ───────────────────────────────────
    if monitoring_active and monitor:
        print_section("Monitor Analysis", "📡")
        if len(monitor.metrics_history) > 0:
            analysis = monitor.analyze_history_with_llm()
            if RICH_AVAILABLE:
                console.print(Panel(json.dumps(analysis, indent=2), title="Monitor Analysis", border_style="cyan"))
            monitor.stop()
        else:
            print_info("No metrics recorded during this session.")
    
    # ── Phase 7: Report ───────────────────────────────────────────────────────
    report_phase(session, monitor if monitoring_active else None)
    
    # ── Farewell ──────────────────────────────────────────────────────────────
    print_section("Session Complete", "🎉")
    if RICH_AVAILABLE:
        console.print("[bold cyan]Thank you for using LLM Doctor! May your models be healthy. ⚕️[/bold cyan]\n")
    else:
        print("\nThank you for using LLM Doctor! ⚕️\n")


if __name__ == "__main__":
    main()
