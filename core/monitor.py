"""
Monitor - Continuous model health monitoring during training and inference.
"""

import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
from pathlib import Path
from core.llm_client import get_json_completion


MONITOR_SYSTEM = """You are an AI model monitoring expert. You analyze metrics from 
training and inference to detect problems in real-time. You look for:
- Loss spikes or divergence
- Gradient explosions/vanishing
- Memory leaks
- Throughput degradation
- Output quality degradation
- Anomalous token distributions

Always provide clear, actionable alerts."""


@dataclass
class MetricSnapshot:
    timestamp: float
    step: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    eval_loss: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class MonitorAlert:
    timestamp: float
    severity: str  # critical, warning, info
    issue: str
    metric_values: dict
    suggested_action: str
    auto_fixable: bool = False
    auto_fix_applied: bool = False
    auto_fix_result: str = ""


class ModelMonitor:
    """
    Continuous monitoring system for LLM training/inference.
    """
    
    def __init__(
        self,
        model_path: str,
        auto_fix: bool = False,
        alert_callback: Optional[Callable] = None,
        check_interval_seconds: float = 10.0,
    ):
        self.model_path = model_path
        self.auto_fix = auto_fix
        self.alert_callback = alert_callback
        self.check_interval = check_interval_seconds
        
        self.metrics_history: list[MetricSnapshot] = []
        self.alerts: list[MonitorAlert] = []
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Thresholds
        self.loss_spike_factor = 3.0     # Alert if loss > 3x recent average
        self.grad_norm_max = 100.0       # Alert if gradient norm > 100
        self.gpu_memory_alert_pct = 0.95 # Alert at 95% GPU memory
        
        # Baseline (computed from first N steps)
        self.baseline_loss: Optional[float] = None
        self.baseline_steps = 20
    
    def push_metrics(self, snapshot: MetricSnapshot):
        """Push a new metrics snapshot for analysis."""
        self.metrics_history.append(snapshot)
        
        # Update baseline
        if len(self.metrics_history) == self.baseline_steps and self.baseline_loss is None:
            recent_losses = [m.loss for m in self.metrics_history if m.loss is not None]
            if recent_losses:
                self.baseline_loss = sum(recent_losses) / len(recent_losses)
        
        # Check for issues
        alerts = self._check_metrics(snapshot)
        for alert in alerts:
            self.alerts.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)
    
    def _check_metrics(self, snapshot: MetricSnapshot) -> list[MonitorAlert]:
        """Check a snapshot for issues and return alerts."""
        alerts = []
        now = snapshot.timestamp
        
        # ── Loss Spike ──────────────────────────────────────────────────────
        if snapshot.loss is not None and self.baseline_loss:
            if snapshot.loss > self.baseline_loss * self.loss_spike_factor:
                alerts.append(MonitorAlert(
                    timestamp=now,
                    severity="critical",
                    issue=f"Loss spike detected: {snapshot.loss:.4f} (baseline: {self.baseline_loss:.4f})",
                    metric_values={"loss": snapshot.loss, "baseline": self.baseline_loss},
                    suggested_action="Reduce learning rate, check data pipeline for corrupt batches, consider checkpoint rollback",
                    auto_fixable=False,
                ))
        
        # ── Loss Divergence ──────────────────────────────────────────────────
        if snapshot.loss is not None and snapshot.loss > 100:
            alerts.append(MonitorAlert(
                timestamp=now,
                severity="critical",
                issue=f"Loss diverged: {snapshot.loss:.4f}",
                metric_values={"loss": snapshot.loss},
                suggested_action="Stop training immediately. Check learning rate, gradient clipping, and data normalization",
                auto_fixable=False,
            ))
        
        # ── Gradient Explosion ────────────────────────────────────────────────
        if snapshot.gradient_norm is not None:
            if snapshot.gradient_norm > self.grad_norm_max:
                alerts.append(MonitorAlert(
                    timestamp=now,
                    severity="critical",
                    issue=f"Gradient explosion: norm={snapshot.gradient_norm:.2f}",
                    metric_values={"gradient_norm": snapshot.gradient_norm},
                    suggested_action="Reduce learning rate or increase gradient clipping threshold",
                    auto_fixable=False,
                ))
            elif snapshot.gradient_norm < 1e-7:
                alerts.append(MonitorAlert(
                    timestamp=now,
                    severity="warning",
                    issue=f"Vanishing gradients: norm={snapshot.gradient_norm:.2e}",
                    metric_values={"gradient_norm": snapshot.gradient_norm},
                    suggested_action="Check model initialization, consider increasing learning rate or different optimizer",
                    auto_fixable=False,
                ))
        
        # ── GPU Memory ────────────────────────────────────────────────────────
        if snapshot.gpu_memory_mb is not None:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    total_mb = float(result.stdout.strip().split("\n")[0])
                    usage_pct = snapshot.gpu_memory_mb / total_mb
                    if usage_pct > self.gpu_memory_alert_pct:
                        alerts.append(MonitorAlert(
                            timestamp=now,
                            severity="warning",
                            issue=f"High GPU memory: {usage_pct*100:.1f}% ({snapshot.gpu_memory_mb:.0f}/{total_mb:.0f} MB)",
                            metric_values={"gpu_memory_mb": snapshot.gpu_memory_mb, "total_mb": total_mb},
                            suggested_action="Reduce batch size, enable gradient checkpointing, or use mixed precision",
                            auto_fixable=False,
                        ))
            except:
                pass
        
        # ── Throughput Drop ───────────────────────────────────────────────────
        if snapshot.throughput_tokens_per_sec is not None and len(self.metrics_history) > 10:
            recent = [m.throughput_tokens_per_sec for m in self.metrics_history[-10:]
                     if m.throughput_tokens_per_sec is not None]
            if recent:
                avg_throughput = sum(recent) / len(recent)
                if snapshot.throughput_tokens_per_sec < avg_throughput * 0.5:
                    alerts.append(MonitorAlert(
                        timestamp=now,
                        severity="warning",
                        issue=f"Throughput dropped 50%: {snapshot.throughput_tokens_per_sec:.0f} tokens/s (avg: {avg_throughput:.0f})",
                        metric_values={"current": snapshot.throughput_tokens_per_sec, "average": avg_throughput},
                        suggested_action="Check GPU utilization, data loading bottleneck, or system resource contention",
                        auto_fixable=False,
                    ))
        
        return alerts
    
    def start_file_monitoring(self, log_file: str):
        """Start monitoring a training log file for metrics."""
        self.is_running = True
        self._stop_event.clear()
        
        def monitor_loop():
            last_size = 0
            while not self._stop_event.is_set():
                try:
                    path = Path(log_file)
                    if path.exists():
                        current_size = path.stat().st_size
                        if current_size > last_size:
                            with open(path) as f:
                                f.seek(last_size)
                                new_lines = f.readlines()
                            last_size = current_size
                            
                            for line in new_lines:
                                metric = self._parse_log_line(line.strip())
                                if metric:
                                    self.push_metrics(metric)
                except Exception as e:
                    pass
                
                self._stop_event.wait(self.check_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop the monitor."""
        self._stop_event.set()
        self.is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _parse_log_line(self, line: str) -> Optional[MetricSnapshot]:
        """Parse a log line for metrics. Supports JSON and common text formats."""
        try:
            # Try JSON format
            data = json.loads(line)
            return MetricSnapshot(
                timestamp=data.get("timestamp", time.time()),
                step=data.get("step"),
                loss=data.get("loss"),
                learning_rate=data.get("lr", data.get("learning_rate")),
                gradient_norm=data.get("grad_norm"),
                gpu_memory_mb=data.get("gpu_memory_mb"),
                throughput_tokens_per_sec=data.get("throughput"),
                eval_loss=data.get("eval_loss"),
            )
        except:
            pass
        
        # Try common text formats: "step=100, loss=2.34, lr=1e-4"
        import re
        snapshot = MetricSnapshot(timestamp=time.time())
        found_any = False
        
        patterns = {
            "step": r'step[=:\s]+(\d+)',
            "loss": r'loss[=:\s]+([\d.eE+-]+)',
            "lr": r'lr[=:\s]+([\d.eE+-]+)',
            "grad_norm": r'grad_norm[=:\s]+([\d.eE+-]+)',
        }
        
        for key, pattern in patterns.items():
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                found_any = True
                val = float(m.group(1))
                if key == "step":
                    snapshot.step = int(val)
                elif key == "loss":
                    snapshot.loss = val
                elif key == "lr":
                    snapshot.learning_rate = val
                elif key == "grad_norm":
                    snapshot.gradient_norm = val
        
        return snapshot if found_any else None
    
    def analyze_history_with_llm(self) -> dict:
        """Use LLM to analyze the full metrics history for patterns."""
        if not self.metrics_history:
            return {"analysis": "No metrics recorded yet"}
        
        # Summarize metrics
        losses = [m.loss for m in self.metrics_history if m.loss is not None]
        grad_norms = [m.gradient_norm for m in self.metrics_history if m.gradient_norm is not None]
        
        summary = {
            "total_steps": len(self.metrics_history),
            "loss_start": losses[0] if losses else None,
            "loss_end": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
        }
        
        messages = [
            {
                "role": "user",
                "content": f"""Analyze this training run metrics summary and identify any patterns or concerns:

{json.dumps(summary, indent=2)}

Alerts raised: {len(self.alerts)}
Alert details: {json.dumps([{"severity": a.severity, "issue": a.issue} for a in self.alerts[:10]], indent=2)}

Return JSON analysis:
{{
  "overall_health": "healthy/warning/critical",
  "training_trend": "converging/diverging/stalled/oscillating",
  "key_observations": ["obs1", "obs2"],
  "recommendations": ["rec1", "rec2"]
}}"""
            }
        ]
        
        return get_json_completion(messages, system_prompt=MONITOR_SYSTEM)
    
    def get_status(self) -> dict:
        """Get current monitor status."""
        latest = self.metrics_history[-1] if self.metrics_history else None
        return {
            "is_running": self.is_running,
            "snapshots_recorded": len(self.metrics_history),
            "total_alerts": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
            "latest_loss": latest.loss if latest else None,
            "latest_step": latest.step if latest else None,
            "recent_alerts": [
                {"severity": a.severity, "issue": a.issue, "time": a.timestamp}
                for a in self.alerts[-5:]
            ],
        }
