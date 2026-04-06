"""Pipeline logger with file output + desktop notifications."""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path


class PipelineLogger:
    """Structured logging with desktop notifications for long-running pipeline jobs.

    Usage:
        logger = PipelineLogger("full_pipeline")
        logger.step_start("IC Evaluation")
        ...
        logger.step_complete("IC Evaluation", "98/770 passed")
        logger.pipeline_complete("4 signals, Sharpe 1.51")
    """

    def __init__(self, run_name: str, log_dir: str = "logs"):
        self._start_time = time.time()
        self._step_start = None
        self._step_name = None

        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{run_name}_{timestamp}.log"

        # Set up Python logger with both file and console handlers
        self.logger = logging.getLogger(f"pipeline.{run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(ch)

        self.logger.info(f"Pipeline started: {run_name}")
        self.logger.info(f"Log file: {log_file}")

    def step_start(self, step_name: str):
        """Log the start of a pipeline step."""
        self._step_name = step_name
        self._step_start = time.time()
        self.logger.info(f">>> {step_name}")

    def step_complete(self, step_name: str, message: str):
        """Log step completion with key metric and elapsed time."""
        elapsed = time.time() - self._step_start if self._step_start else 0
        total = time.time() - self._start_time
        msg = f"  {step_name}: {message} [{_fmt_time(elapsed)}, total {_fmt_time(total)}]"
        self.logger.info(msg)
        self._notify(f"{step_name}: {message} ({_fmt_time(elapsed)})")
        self._step_start = None

    def step_failed(self, step_name: str, error: str):
        """Log step failure."""
        elapsed = time.time() - self._step_start if self._step_start else 0
        msg = f"  FAILED {step_name}: {error} [{_fmt_time(elapsed)}]"
        self.logger.error(msg)
        self._notify(f"FAILED {step_name}: {error}", urgency="critical")

    def stepwise_update(self, step: int, signal: str, sharpe: float, oos_sharpe: float):
        """Log a stepwise selection step."""
        elapsed = time.time() - self._step_start if self._step_start else 0
        msg = f"  Stepwise {step}: +{signal} -> SR={sharpe:.4f} OOS={oos_sharpe:.4f} [{_fmt_time(elapsed)}]"
        self.logger.info(msg)
        self._notify(f"Stepwise {step}: +{signal} SR={sharpe:.3f}")

    def pipeline_complete(self, summary: str):
        """Log pipeline completion."""
        total = time.time() - self._start_time
        msg = f"COMPLETE: {summary} [total {_fmt_time(total)}]"
        self.logger.info(msg)
        self._notify(f"Pipeline Complete: {summary} ({_fmt_time(total)})", urgency="normal")

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(f"  {message}")

    def _notify(self, message: str, urgency: str = "normal"):
        """Send desktop notification via notify-send."""
        try:
            subprocess.run(
                ["notify-send", "-u", urgency, "-t", "5000", "Alpha Pipeline", message],
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # notify-send not available or hung


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
