import json
from pathlib import Path

CACHE_FILE = Path(r"C:\Project\webapp\backend\cache\doc_cache.json")

GENERATOR_KEYS = [
    "c806ec8c17babc3cb6ccd413a1971904",
    "fe9e667c9426a14325fb24759c6379f0",
    "b28f9cc1b9a5926a939ac9148fbd7b08",
    "098a61d181f2a7b282c828fb4146642d",
    "b65c383a7083f93fb9b37ae151801a1f",
    "39bc2d376d358daed1a4e0dce7b3e02d",
    "617a9c27687bd758bb2250ea670b0bf8",
    "a9ba1b194f1d6c6b9ed7488edd1fceb7",
    "2a4d8b545416e5c64f19a1e74419e82b",
    "90177814a3eab80654d0e9162e76f85b",
    "67d05986f3b9a96d05b03189a574391f",
    "79e3eb9e6059d02af67666d2ccbd5e1f",
    "44fbfb9860217b888488a7e52ab87077",
    "d8606d1f11a1a6ed7ff7552707e1389c",
    "a8284ab14e791467b918b4b12ed5c80d",
    "79aecfb311fd3ab76e6ea757266c7a34",
    "d68650ff395577be971ab51a1c7602b2",
    "a5136d2329562219cca6e3edd212b499",
]

data = json.loads(CACHE_FILE.read_text())
for k in GENERATOR_KEYS:
    data.pop(k, None)
CACHE_FILE.write_text(json.dumps(data, indent=2))
print(f"✅ Deleted {len(GENERATOR_KEYS)} Generator.java cache entries")