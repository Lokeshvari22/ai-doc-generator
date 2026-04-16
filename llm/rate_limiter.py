"""
rate_limiter.py — RPM + RPD tracking with smart backoff.

FIX: with_backoff() now distinguishes two kinds of 429:
  - Daily quota exhausted  → raise immediately (DailyQuotaExhausted),
                             retrying is pointless until tomorrow.
  - RPM window hit         → wait 65 s to clear the sliding window,
                             then retry.
FIX: retryDelay from the API error body is parsed and respected when present.
"""

import re
import time
import random
from collections import deque
from datetime import datetime, timedelta


# Sentinel so callers can catch daily-quota separately from other errors
class DailyQuotaExhausted(Exception):
    """Raised when the API returns a per-day quota violation."""
    pass


def _is_daily_quota(msg: str) -> bool:
    """
    Return True when the error is a hard daily cap, not a per-minute limit.
    Gemini signals this with quotaId containing 'PerDay' or 'perday',
    or the phrase 'per day' / 'daily' in the message body.
    """
    m = msg.lower()
    return (
        "perday" in m
        or "per day" in m
        or "per_day" in m
        or "requestsperday" in m
        or "generaterequest" in m and "day" in m   # GenerateRequestsPerDayPerProject
    )


def _parse_retry_delay(msg: str) -> float | None:
    """
    Extract the server-suggested retry delay in seconds from the error message.
    Gemini returns: 'Please retry in 32.557389757s'
    Returns None if not found.
    """
    match = re.search(r"retry(?:\s+in)?\s+([\d.]+)\s*s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


class RateLimiter:
    """Tracks BOTH RPM and RPD limits. Automatically waits when close to limits."""

    def __init__(self, rpm: int, name: str, rpd: int = 999_999):
        self.rpm      = int(rpm * 0.85)   # 85% safe margin
        self.rpd      = rpd
        self.name     = name
        self.requests = deque()            # sliding RPM window

        self.day_count = 0
        self.day_reset = datetime.now() + timedelta(days=1)

    def wait(self):
        if datetime.now() >= self.day_reset:
            self.day_count = 0
            self.day_reset = datetime.now() + timedelta(days=1)

        if self.day_count >= int(self.rpd * 0.95):
            raise DailyQuotaExhausted(
                f"[{self.name}] Daily limit reached "
                f"({self.day_count}/{self.rpd}). Try again tomorrow."
            )

        now    = time.time()
        cutoff = now - 60
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        if len(self.requests) >= self.rpm:
            oldest = self.requests[0]
            wait   = 60 - (time.time() - oldest) + 1
            wait  += random.uniform(0.1, 0.5)
            print(f"  [{self.name}] RPM limit, waiting {wait:.1f}s...")
            time.sleep(wait)

        self.requests.append(time.time())
        self.day_count += 1

    def seconds_until_slot(self) -> float:
        if len(self.requests) < self.rpm:
            return 0.0
        oldest = self.requests[0]
        return max(0.0, 60 - (time.time() - oldest) + 1)

    def status(self) -> dict:
        now    = time.time()
        cutoff = now - 60
        recent = sum(1 for r in self.requests if r > cutoff)
        return {
            "name"         : self.name,
            "rpm_used"     : recent,
            "rpm_limit"    : self.rpm,
            "day_used"     : self.day_count,
            "day_limit"    : self.rpd,
            "day_remaining": self.rpd - self.day_count,
        }


def with_backoff(func, retries: int = 4):
    """
    Retry with smart backoff on rate-limit (429) and transient errors.

    Two 429 cases handled differently:
      Daily quota exhausted  → raise DailyQuotaExhausted immediately.
                               Waiting won't help — quota resets at midnight.
      RPM window hit         → wait for the server-suggested retryDelay
                               (or 65 s if none provided) then retry.

    Wait schedule on RPM 429 (no retryDelay in error):
      attempt 0 → 65 s
      attempt 1 → 80 s
      attempt 2 → 95 s
      attempt 3 → raise

    Wait schedule on other errors:
      2 s between attempts (transient network issues)
    """
    for attempt in range(retries):
        try:
            return func()

        except DailyQuotaExhausted:
            # Already the right type — propagate immediately
            raise

        except Exception as e:
            msg = str(e)

            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():

                # ── Daily cap hit → fail fast, no point retrying ──────────
                if _is_daily_quota(msg):
                    raise DailyQuotaExhausted(
                        f"Daily quota exhausted. Original error:\n{msg}"
                    ) from e

                # ── RPM hit → wait and retry ──────────────────────────────
                if attempt >= retries - 1:
                    raise

                # Respect server-suggested delay; otherwise use default ramp
                suggested = _parse_retry_delay(msg)
                if suggested and suggested > 0:
                    wait = suggested + random.uniform(2.0, 5.0)   # small buffer
                else:
                    wait = 65 + attempt * 15 + random.uniform(1.0, 4.0)

                print(
                    f"  Rate limited (RPM), waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{retries})..."
                )
                time.sleep(wait)

            elif attempt < retries - 1:
                time.sleep(2 + random.uniform(0, 1))
            else:
                raise

    raise Exception("Max retries exceeded")