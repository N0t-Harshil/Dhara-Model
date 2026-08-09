"""Check if bigcode/the-stack-v2-dedup exists and what format it uses."""
import requests

paths = ["bigcode/the-stack-v2-dedup", "bigcode/the-stack-v2"]
for p in paths:
    try:
        r = requests.get(f"https://huggingface.co/api/datasets/{p}", timeout=15)
        if r.status_code != 200:
            print(f"{p}: HTTP {r.status_code}")
            continue
        data = r.json()
        print(f"{p}: EXISTS")
        configs = data.get("configs", [])
        print(f"  configs: {len(configs)}")
        for c in configs[:3]:
            cn = c.get("config", {}).get("name", "?")
            print(f"    - {cn}")
        # Check for parquet files
        try:
            r2 = requests.get(f"https://huggingface.co/api/datasets/{p}/tree/main/data", timeout=10)
            if r2.status_code == 200:
                files = [f["path"] for f in r2.json() if f["path"].endswith(".parquet")]
                print(f"  parquet files: {len(files)}")
            else:
                print(f"  /data/ tree: HTTP {r2.status_code}")
                # Try without /data/
                r3 = requests.get(f"https://huggingface.co/api/datasets/{p}/tree/main", timeout=10)
                if r3.status_code == 200:
                    items = [f["path"] for f in r3.json()]
                    print(f"  root items: {items[:10]}")
        except Exception as e:
            print(f"  tree error: {e}")
    except Exception as e:
        print(f"{p}: ERROR {e}")
