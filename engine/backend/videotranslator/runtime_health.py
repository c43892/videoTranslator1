"""Activity-based supervision. No elapsed-runtime limit for healthy work."""
import os
from pathlib import Path
import subprocess
import time


class ProcessActivity:
    """Observe CPU work or substantial I/O across a Linux process tree."""
    def __init__(self, pid, proc_root='/proc'):
        self.pid, self.root = pid, Path(proc_root)
        self.previous = {}
        self.ticks = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100

    def sample(self):
        current, pending = {}, [self.pid]
        while pending:
            pid = pending.pop()
            directory = self.root / str(pid)
            try:
                fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
                io = dict(line.split(':', 1) for line in (directory / 'io').read_text().splitlines())
                current[(pid, fields[19])] = (int(fields[11]) + int(fields[12]),
                                             int(io.get('rchar', 0)) + int(io.get('wchar', 0)))
                # Children may be created from any thread (e.g. Demucs' background worker).
                for children in directory.glob('task/*/children'):
                    pending.extend(int(value) for value in children.read_text().split())
            except (OSError, ValueError, IndexError):
                continue  # A child may exit during sampling.
        active = False
        for key, counters in current.items():
            prior = self.previous.get(key)
            if prior is not None and (counters[0] - prior[0] >= self.ticks * .1 or
                                      counters[1] - prior[1] >= 65536):
                active = True
            elif prior is None and self.previous:
                active = True
        self.previous = current
        return active


def gpu_active():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3, check=True)
        return any(float(value.strip()) > 0 for value in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


class ActivityWatchdog:
    def __init__(self, idle_seconds=900, clock=time.monotonic):
        self.clock, self.idle_seconds = clock, idle_seconds
        self.last_activity, self.signature = clock(), None

    def observe(self, signature, active=False):
        if active or signature != self.signature:
            self.last_activity = self.clock()
        self.signature = signature
        return self.clock() - self.last_activity < self.idle_seconds
