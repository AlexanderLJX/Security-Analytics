import argparse, csv, json, math, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

CURL = "curl.exe" if sys.platform.startswith("win") else "curl"

def curl_json_get(url: str, timeout: int = 5):
    args = [CURL, "-sS", "--fail-with-body", "-H", "Accept: application/json", url]
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout)
        if p.returncode != 0:
            return None
        return json.loads(p.stdout.decode("utf-8", errors="ignore"))
    except Exception:
        return None

def curl_json_post(url: str, payload: dict, timeout: int = 600):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    args = [CURL, "-sS", "--fail-with-body", "-X", "POST",
            "-H", "Content-Type: application/json",
            "--data-binary", "@-", url]
    try:
        p = subprocess.run(args, input=body, capture_output=True, timeout=timeout)
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", errors="ignore") or p.stdout.decode("utf-8", errors="ignore")
            return {"ok": False, "error": err.strip()}
        out = p.stdout.decode("utf-8", errors="ignore")
        return {"ok": True, "data": json.loads(out)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def load_emails(csv_path: str, column: str = "text", limit: int | None = None, skip_rows: int = 0) -> List[str]:
    emails = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise SystemExit(f"CSV must have a '{column}' column. Found: {reader.fieldnames}")
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue
            txt = (row.get(column) or "").strip()
            if txt:
                emails.append(txt)
            if limit and len(emails) >= limit:
                break
    return emails

def run_analyze(base_url: str, emails: List[str], out_path: str, workers: int, timeout: int, include_text: bool, retries: int):
    url = f"{base_url.rstrip('/')}/analyze"
    def send_one(idx_email):
        idx, email = idx_email
        last_err = None
        for attempt in range(retries + 1):
            res = curl_json_post(url, {"email": email}, timeout=timeout)
            if res.get("ok"):
                payload = {"index": idx, "result": res["data"]}
                if include_text: payload["email"] = email
                return True, payload
            last_err = res.get("error", "unknown error")
            time.sleep(1.5 ** attempt)
        payload = {"index": idx, "error": last_err}
        if include_text: payload["email"] = email
        return False, payload

    start = time.time()
    ok = err = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        if workers <= 1:
            for idx, email in enumerate(emails):
                success, payload = send_one((idx, email))
                ok += int(success); err += int(not success)
                fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx+1}/{len(emails)} (ok={ok}, err={err})")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(send_one, (i, e)): i for i, e in enumerate(emails)}
                for count, fut in enumerate(as_completed(futures), 1):
                    success, payload = fut.result()
                    ok += int(success); err += int(not success)
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    if count % 50 == 0:
                        print(f"Processed {count}/{len(emails)} (ok={ok}, err={err})")
    print(f"Done analyze: {ok} ok, {err} errors in {time.time()-start:.1f}s -> {out_path}")

def run_batch(base_url: str, emails: List[str], out_path: str, chunk_size: int, timeout: int, include_text: bool):
    url = f"{base_url.rstrip('/')}/batch"
    start = time.time()
    ok = err = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in range(0, len(emails), chunk_size):
            chunk = emails[i:i+chunk_size]
            res = curl_json_post(url, {"emails": chunk}, timeout=timeout)
            if not res.get("ok"):
                for j, email in enumerate(chunk):
                    payload = {"index": i + j, "error": f"batch_failed: {res.get('error','unknown')}"}
                    if include_text: payload["email"] = email
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    err += 1
            else:
                results = res["data"].get("results", [])
                for j, item in enumerate(results):
                    idx = i + j
                    if isinstance(item, dict) and "error" in item:
                        payload = {"index": idx, "error": item["error"]}
                        err += 1
                    else:
                        payload = {"index": idx, "result": item}
                        ok += 1
                    if include_text: payload["email"] = chunk[j]
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"Processed {min(i+chunk_size, len(emails))}/{len(emails)} (ok={ok}, err={err})")
    print(f"Done batch: {ok} ok, {err} errors in {time.time()-start:.1f}s -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Upload CSV to inference server via curl")
    ap.add_argument("--file", default="data/combined-dataset-sample-1000x.csv")
    ap.add_argument("--base-url", default="http://localhost:5000")
    ap.add_argument("--mode", choices=["analyze", "batch"], default="analyze")
    ap.add_argument("--workers", type=int, default=1)  # keep 1 for GGUF safety
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--batch-timeout", type=int, default=1800)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--include-text", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-rows", type=int, default=0)
    args = ap.parse_args()

    health = curl_json_get(f"{args.base_url.rstrip('/')}/health", timeout=5)
    if health:
        print(f"Server healthy. Stats: {health.get('stats', {})}")
    else:
        print("Warning: health check failed. Ensure server is running.")

    emails = load_emails(args.file, limit=args.limit, skip_rows=args.skip_rows)
    if not emails:
        raise SystemExit("No emails loaded from CSV.")
    print(f"Loaded {len(emails)} emails from {args.file}")

    if args.mode == "analyze":
        run_analyze(args.base_url, emails, args.out, args.workers, args.timeout, args.include_text, args.retries)
    else:
        run_batch(args.base_url, emails, args.out, args.chunk_size, args.batch_timeout, args.include_text)

if __name__ == "__main__":
    main()
