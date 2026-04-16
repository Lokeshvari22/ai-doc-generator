"""
check_groq_gemini_limits.py
Run: python check_groq_gemini_limits.py
Shows rate limits for Groq and Gemini models.
"""

import requests
import time

# ── Paste your API keys here ───────────────────────────────────────────────────
GROQ_API_KEY   = "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"    # from console.groq.com
GEMINI_API_KEY = "AIzaSyC9Moz4PH0lxg3ySerdltgtpc3bPhH34KE"  # from aistudio.google.com
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
]

GEMINI_MODELS = [
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# ── CHECK GROQ ────────────────────────────────────────────────────────────────
print("=" * 78)
print("🟠 GROQ MODELS")
print("=" * 78)
print(f"{'Model':<40} {'TPM':>10} {'RPM':>8} {'RPD':>8}  Status")
print("-" * 78)

for model in GROQ_MODELS:
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            },
            timeout=15
        )

        h = response.headers

        if response.status_code == 200:
            tpm = h.get("x-ratelimit-limit-tokens",           "N/A")  # tokens per minute
            rpm = h.get("x-ratelimit-limit-requests",         "N/A")  # requests per minute
            rpd = h.get("x-ratelimit-limit-requests-per-day", "N/A")  # requests per day
            print(f"{model:<40} {str(tpm):>10} {str(rpm):>8} {str(rpd):>8}  ✅")
        elif response.status_code == 429:
            print(f"{model:<40} {'N/A':>10} {'N/A':>8} {'N/A':>8}  ⏳ 429")
        elif response.status_code == 404:
            print(f"{model:<40} {'N/A':>10} {'N/A':>8} {'N/A':>8}  ❌ 404")
        else:
            print(f"{model:<40} {'N/A':>10} {'N/A':>8} {'N/A':>8}  ⚠️ {response.status_code}")

        time.sleep(0.5)

    except Exception as e:
        print(f"{model:<40} {'N/A':>10} {'N/A':>8} {'N/A':>8}  ❌ Err")


# ── CHECK GEMINI ──────────────────────────────────────────────────────────────
print("\n")
print("=" * 78)
print("🔵 GEMINI MODELS")
print("=" * 78)
print(f"{'Model':<40} {'RPM':>8} {'TPM':>12} {'RPD':>8}  Status")
print("-" * 78)

for model in GEMINI_MODELS:
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": "Hi"}]}],
                "generationConfig": {"maxOutputTokens": 5}
            },
            timeout=15
        )

        h = response.headers

        if response.status_code == 200:
            rpm = h.get("x-ratelimit-limit",              "N/A")
            tpm = h.get("x-ratelimit-limit-tokens",       "N/A")
            rpd = h.get("x-ratelimit-limit-requests-day", "N/A")
            print(f"{model:<40} {str(rpm):>8} {str(tpm):>12} {str(rpd):>8}  ✅")
        elif response.status_code == 429:
            print(f"{model:<40} {'N/A':>8} {'N/A':>12} {'N/A':>8}  ⏳ 429")
        elif response.status_code == 404:
            print(f"{model:<40} {'N/A':>8} {'N/A':>12} {'N/A':>8}  ❌ 404")
        else:
            print(f"{model:<40} {'N/A':>8} {'N/A':>12} {'N/A':>8}  ⚠️ {response.status_code}")

        time.sleep(0.5)

    except Exception as e:
        print(f"{model:<40} {'N/A':>8} {'N/A':>12} {'N/A':>8}  ❌ Err")

print("\n✅ Done!")
print("\n📌 Notes:")
print("   Groq  TPM = Tokens Per Minute  | RPM = Requests Per Minute | RPD = Requests Per Day")
print("   Gemini limits are per-model and may not appear in headers — check AI Studio instead")
print("   👉 Groq limits  : https://console.groq.com/settings/limits")
print("   👉 Gemini limits: https://aistudio.google.com/app/apikey")