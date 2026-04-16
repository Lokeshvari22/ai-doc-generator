# Run this FIRST before main.py
# python check_keys.py

import os
import sys
import requests
from groq import Groq
from google import genai


# ── Load .env manually ────────────────────────────────────
def load_env():
    if not os.path.exists(".env"):
        print("⚠️  No .env file found!")
        print("   Create .env with your API keys")
        return
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") \
                    and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()
    print("✅ .env loaded\n")


# ── Check keys exist ──────────────────────────────────────
def check_keys_exist():
    print("=" * 50)
    print("  Step 1: Checking Keys Exist")
    print("=" * 50)

    keys = {
        "GROQ_API_KEY"  : ("gsk_",),
        "GEMINI_API_KEY": ("AIza",),
        "GITHUB_TOKEN"  : ("ghp_", "github_pat_"),
    }

    all_ok = True
    for name, prefixes in keys.items():
        val = os.getenv(name)
        if not val:
            print(f"  ❌ {name}: NOT SET")
            all_ok = False
        elif not any(val.startswith(p) for p in prefixes):
            print(f"  ⚠️  {name}: Set but prefix unexpected")
            print(f"      Value starts: {val[:12]}...")
            all_ok = False
        else:
            masked = val[:10] + "*" * 6
            print(f"  ✅ {name}: {masked}")

    print()
    return all_ok


# ── Test Groq ─────────────────────────────────────────────
def test_groq():
    print("Testing Groq (llama-3.1-8b-instant)...")
    try:
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [{"role": "user",
                         "content": "Reply with one word: OK"}],
            max_tokens = 5,
            temperature = 0
        )
        reply = response.choices[0].message.content.strip()
        print(f"  ✅ Groq works → Response: {reply}\n")
        return True
    except Exception as e:
        print(f"  ❌ Groq failed → {e}\n")
        return False


# ── Test Gemini 2.5 ───────────────────────────────────────
def test_gemini():
    print("Testing Gemini (gemini-2.5-flash)...")
    try:
        client   = genai.Client(
            api_key = os.getenv("GEMINI_API_KEY")
        )
        response = client.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = "Reply with one word: OK"
        )
        reply = response.text.strip()
        print(f"  ✅ Gemini works → Response: {reply}\n")
        return True
    except Exception as e:
        print(f"  ❌ Gemini failed → {e}\n")
        return False


# ── Test GitHub ───────────────────────────────────────────
def test_github():
    print("Testing GitHub (openai/gpt-4o-mini)...")
    try:
        token = os.getenv("GITHUB_TOKEN")
        url   = ("https://models.github.ai"
                 "/inference/chat/completions")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type" : "application/json"
        }
        payload = {
            "model"      : "openai/gpt-4o-mini",
            "messages"   : [{"role": "user",
                             "content": "Reply with one word: OK"}],
            "max_tokens" : 5,
            "temperature": 0
        }
        resp = requests.post(
            url, headers=headers,
            json=payload, timeout=15
        )
        if resp.status_code == 200:
            reply = (resp.json()
                     ["choices"][0]["message"]["content"])
            print(f"  ✅ GitHub works → Response: {reply}\n")
            return True
        else:
            print(f"  ❌ GitHub failed → "
                  f"{resp.status_code}: {resp.text}\n")
            return False
    except Exception as e:
        print(f"  ❌ GitHub failed → {e}\n")
        return False


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    load_env()

    keys_ok = check_keys_exist()
    if not keys_ok:
        print("Fix keys in .env then run again")
        sys.exit(1)

    print("=" * 50)
    print("  Step 2: Live API Tests")
    print("=" * 50)

    results = {
        "Groq"  : test_groq(),
        "Gemini": test_gemini(),
        "GitHub": test_github(),
    }

    print("=" * 50)
    print("  Final Result")
    print("=" * 50)
    for api, ok in results.items():
        icon = "✅ Ready" if ok else "❌ Fix this"
        print(f"  {api:10} {icon}")

    if all(results.values()):
        print("\n  🚀 All good! Run: python main.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  Fix these: {', '.join(failed)}")