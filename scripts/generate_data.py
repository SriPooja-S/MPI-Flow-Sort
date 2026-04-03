#!/usr/bin/env python3
"""
generate_data.py  –  Generate binary input files for the distributed sort system.

LOW-RAM / LOW-DISK DESIGN:
  • Uses a write buffer of at most WRITE_BATCH records at a time (default 8192 = ~800 KB).
  • The duplicate-key mode keeps only a SMALL fixed pool (default 1000 keys) in RAM,
    not a pool proportional to record count.  This was the cause of OOM on large files.
  • Checks available disk space before writing and aborts early with a clear message.
  • Progress is shown every 500k records so you can Ctrl-C if disk fills up.
  • The --max-mb flag caps output size regardless of --count.

Usage:
    # 10 million records  (~954 MB)  – random
    python3 generate_data.py --count 10000000 --output data.bin

    # 100 million records (~9.3 GB)  – SKEWED distribution  (fixed from v1 OOM)
    python3 generate_data.py --count 100000000 --skew --output skewed.bin

    # 10 million records with ~50% duplicate keys
    python3 generate_data.py --count 10000000 --dups 0.5 --output dupes.bin

    # Cap output at 500 MB regardless of --count
    python3 generate_data.py --count 999999999 --max-mb 500 --output data.bin

    # Verify an existing sorted file
    python3 generate_data.py --verify sorted.bin
"""

import argparse
import os
import struct
import random
import sys
import time
import shutil

RECORD_SIZE  = 100
KEY_SIZE     = 10
PAYLOAD_SIZE = 90

# Maximum records in the write buffer at once  (8192 × 100 B = 800 KB)
WRITE_BATCH  = 8192

# Fixed-size key pool for duplicate mode  (never proportional to record count)
DUP_POOL_SIZE = 1000   # 1000 × 10 B = 10 KB RAM – safe on any machine


# ── Disk space guard ──────────────────────────────────────────────────────────
def check_disk_space(path, needed_bytes, label="output"):
    try:
        total, used, free = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
        if free < needed_bytes:
            free_mb   = free / (1024**2)
            needed_mb = needed_bytes / (1024**2)
            print(f"\nERROR: Not enough disk space for {label}.")
            print(f"  Available : {free_mb:.1f} MB")
            print(f"  Required  : {needed_mb:.1f} MB")
            print(f"  Tip       : Use --max-mb to cap output size, or choose a different "
                  f"--output path on a larger partition.")
            return False
    except Exception:
        pass   # can't check – proceed
    return True


# ── Key generators (no large allocations) ─────────────────────────────────────
def make_random_key(rng):
    return rng.randbytes(KEY_SIZE)

def make_skewed_key(rng):
    """80% of keys in the lowest 20% of the range."""
    if rng.random() < 0.80:
        return bytes([rng.randint(0, 50)] + [rng.randint(0, 255) for _ in range(KEY_SIZE-1)])
    return rng.randbytes(KEY_SIZE)

def make_sorted_key(i, total, rng):
    """Monotonically increasing with small noise – simulates nearly-sorted input."""
    base  = int(i * ((2**80) / max(total, 1)))
    noise = rng.randint(0, 1000)
    val   = max(0, min(base + noise, 2**80 - 1))
    return val.to_bytes(KEY_SIZE, 'big')


# ── Main generator ────────────────────────────────────────────────────────────
def generate_data(count, output, seed=42, skew=False, presorted=False,
                  dup_fraction=0.0, max_mb=0):
    rng = random.Random(seed)

    # Apply --max-mb cap
    max_records = count
    if max_mb > 0:
        cap = int(max_mb * 1024 * 1024 / RECORD_SIZE)
        if cap < count:
            print(f"Note: capping output at {cap:,} records ({max_mb} MB).")
            max_records = cap

    total_bytes = max_records * RECORD_SIZE

    if not check_disk_space(output, total_bytes):
        sys.exit(1)

    print(f"Generating {max_records:,} records  →  {total_bytes/(1024**3):.3f} GB  →  {output}")
    if dup_fraction > 0:
        print(f"  Duplicate fraction: {dup_fraction:.0%}  "
              f"(fixed pool of {DUP_POOL_SIZE} keys – low RAM)")
    if skew:
        print("  Key distribution: skewed (80% of keys in lowest 20% of range)")

    # Pre-generate the SMALL fixed dup pool (only if needed)
    dup_pool = []
    if dup_fraction > 0:
        dup_pool = [rng.randbytes(KEY_SIZE) for _ in range(DUP_POOL_SIZE)]

    t0 = time.time()
    batch = bytearray()    # reused write buffer

    with open(output, 'wb') as f:
        written = 0
        report_every = 500_000   # progress every 500k records

        for i in range(max_records):
            # ── Key ─────────────────────────────────────────────────────────
            if dup_fraction > 0 and rng.random() < dup_fraction:
                # Pick from small fixed pool – O(1) RAM, no OOM
                key = dup_pool[rng.randint(0, DUP_POOL_SIZE - 1)]
            elif skew:
                key = make_skewed_key(rng)
            elif presorted:
                key = make_sorted_key(i, max_records, rng)
            else:
                key = make_random_key(rng)

            # ── Payload ─────────────────────────────────────────────────────
            payload = rng.randbytes(PAYLOAD_SIZE)

            batch += key + payload

            # Flush when buffer is full (every WRITE_BATCH records)
            if len(batch) >= WRITE_BATCH * RECORD_SIZE:
                f.write(batch)
                written += WRITE_BATCH
                batch = bytearray()

                # Progress report
                if written % report_every < WRITE_BATCH:
                    pct     = written / max_records * 100
                    elapsed = time.time() - t0
                    mb_s    = written * RECORD_SIZE / (1024**2) / max(elapsed, 1e-6)
                    print(f"  {pct:5.1f}%  {written:,} records  {mb_s:.0f} MB/s", end='\r')

        # Flush remainder
        if batch:
            f.write(batch)
            written += len(batch) // RECORD_SIZE

    elapsed   = time.time() - t0
    size_mb   = os.path.getsize(output) / (1024**2)
    mb_s      = size_mb / max(elapsed, 1e-6)
    print(f"\nDone: {written:,} records  {size_mb:.1f} MB  in {elapsed:.2f}s  ({mb_s:.0f} MB/s)")


# ── Streaming verifier ────────────────────────────────────────────────────────
def verify_sorted(path):
    """
    Stream through file checking ordering.  Reads in 64 KB pages – no full load.
    """
    file_size = os.path.getsize(path)
    if file_size % RECORD_SIZE != 0:
        print(f"ERROR: file size {file_size} not a multiple of {RECORD_SIZE}")
        return
    total = file_size // RECORD_SIZE
    print(f"Verifying {path}  ({total:,} records, {file_size/(1024**2):.1f} MB) ...")

    PAGE      = 640    # 640 × 100 B = 64 KB per read
    prev      = None
    count     = 0
    violations= 0

    with open(path, 'rb') as f:
        while True:
            chunk = f.read(PAGE * RECORD_SIZE)
            if not chunk:
                break
            for off in range(0, len(chunk), RECORD_SIZE):
                rec = chunk[off:off + RECORD_SIZE]
                if len(rec) < RECORD_SIZE:
                    break
                key = rec[:KEY_SIZE]
                if prev is not None and key < prev:
                    violations += 1
                    if violations <= 3:
                        print(f"  Violation at record {count}: {key.hex()} < {prev.hex()}")
                prev   = key
                count += 1

    if violations == 0:
        print(f"  PASSED: {count:,} records correctly sorted")
    else:
        print(f"  FAILED: {violations} ordering violations in {count:,} records")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate / verify binary test data for distributed sort",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--count',   type=int,   default=1_000_000,
                    help='Number of records to generate (default: 1 000 000)')
    ap.add_argument('--output',  type=str,   default='data.bin',
                    help='Output file path (default: data.bin)')
    ap.add_argument('--seed',    type=int,   default=42,
                    help='Random seed (default: 42)')
    ap.add_argument('--skew',    action='store_true',
                    help='Generate skewed key distribution (80/20 rule)')
    ap.add_argument('--sorted',  action='store_true',
                    help='Generate nearly pre-sorted data')
    ap.add_argument('--dups',    type=float, default=0.0,
                    help='Fraction of records that reuse a duplicate key [0.0–1.0]')
    ap.add_argument('--max-mb',  type=int,   default=0,
                    help='Cap output at this many MB (0 = no cap)')
    ap.add_argument('--verify',  type=str,   default='',
                    help='Verify an existing sorted file instead of generating')
    args = ap.parse_args()

    if args.verify:
        verify_sorted(args.verify)
        return

    if args.dups < 0.0 or args.dups > 1.0:
        print("ERROR: --dups must be between 0.0 and 1.0"); sys.exit(1)

    generate_data(
        count        = args.count,
        output       = args.output,
        seed         = args.seed,
        skew         = args.skew,
        presorted    = args.sorted,
        dup_fraction = args.dups,
        max_mb       = args.max_mb,
    )


if __name__ == '__main__':
    main()
