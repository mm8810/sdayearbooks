
# rate_limiter.py
import time, random, threading

class TokenBucket:
    """
    Limiter for BOTH tokens/min (TPM) and requests/min (RPM).
    Thread-safe and continuous refill.
    """
    def __init__(self, tokens_per_min: int, requests_per_min: int):
        self.TPM = max(1, int(tokens_per_min))
        self.RPM = max(1, int(requests_per_min))
        self._lock = threading.Lock()
        self._tokens = float(self.TPM)
        self._reqs   = float(self.RPM)
        self._last   = time.time()

    def _refill(self):
        now = time.time()
        dt = now - self._last
        if dt <= 0:
            return
        self._tokens = min(self.TPM, self._tokens + (self.TPM/60.0)*dt)
        self._reqs   = min(self.RPM, self._reqs   + (self.RPM/60.0)*dt)
        self._last = now

    def acquire(self, tokens_needed: int):
        tokens_needed = max(1, int(tokens_needed))
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens_needed and self._reqs >= 1.0:
                    self._tokens -= tokens_needed
                    self._reqs   -= 1.0
                    return
                token_def = max(0.0, tokens_needed - self._tokens)
                req_def   = max(0.0, 1.0 - self._reqs)
                wait_token = token_def / (self.TPM/60.0) if token_def > 0 else 0.0
                wait_req   = req_def   / (self.RPM/60.0) if req_def   > 0 else 0.0
                wait_s = max(wait_token, wait_req, 0.05)
            time.sleep(wait_s)

def estimate_tokens_from_text(text: str) -> int:
    """
    ~4 characters per token heuristic to avoid external tokenizers.
    """
    if text is None:
        return 1
    return max(1, len(text) // 4)
