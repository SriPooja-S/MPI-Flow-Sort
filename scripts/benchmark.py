#!/usr/bin/env python3
"""
benchmark.py – Automated performance benchmarking for the distributed sort system.
Low-RAM aware: generates smaller default datasets and uses -m 64 by default.

Usage:
    # Auto-generate 2M records and benchmark np=1,2
    python3 benchmark.py --binary ./distributed_sort --generate 2000000

    # Use existing file
    python3 benchmark.py --binary ./distributed_sort --input data.bin --np-list 1,2,4

    # For a 2 GB RAM machine – conservative settings
    python3 benchmark.py --binary ./distributed_sort --generate 1000000 --mem 64 --np-list 1,2
"""

import argparse
import subprocess
import time
import os
import sys
import json
import shutil

RECORD_SIZE = 100

def check_disk(path, needed_mb):
    try:
        _, _, free = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) or ".")
        if free < needed_mb * 1024 * 1024:
            print(f"WARNING: Only {free/(1024**2):.0f} MB free, need ~{needed_mb} MB")
    except Exception:
        pass

def format_size(n_bytes):
    for u in ['B','KB','MB','GB']:
        if n_bytes < 1024: return f"{n_bytes:.1f} {u}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"

def run_sort(binary, inp, out, np, mem_mb=64, verify=False):
    cmd = ['mpirun', '-np', str(np), binary,
           '-i', inp, '-o', out, '-m', str(mem_mb), '-b', '512']
    if verify: cmd.append('--verify')
    t0 = time.perf_counter()
    r  = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'np': np, 'elapsed': time.perf_counter()-t0,
        'success': r.returncode == 0,
        'stdout': r.stdout, 'stderr': r.stderr,
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark distributed sort (low-RAM aware)")
    ap.add_argument('--binary',   default='./distributed_sort')
    ap.add_argument('--input',    default='')
    ap.add_argument('--generate', type=int, default=0,
                    help='Generate N records before benchmarking')
    ap.add_argument('--np-list',  default='1,2',
                    help='Comma-separated MPI process counts (default: 1,2)')
    ap.add_argument('--mem',      type=int, default=64,
                    help='Per-node RAM limit in MB (default: 64 – safe for 2 GB systems)')
    ap.add_argument('--output',   default='/tmp/bench_out.bin')
    ap.add_argument('--verify',   action='store_true')
    ap.add_argument('--json',     default='')
    args = ap.parse_args()

    # Generate data
    inp = args.input
    if args.generate > 0:
        inp = '/tmp/bench_input.bin'
        needed_mb = args.generate * RECORD_SIZE // (1024*1024) * 3  # 3× for temp files
        check_disk(inp, needed_mb)
        print(f"Generating {args.generate:,} records...")
        gen = os.path.join(os.path.dirname(__file__), 'generate_data.py')
        subprocess.run([sys.executable, gen,
                        '--count', str(args.generate), '--output', inp], check=True)

    if not inp or not os.path.exists(inp):
        print("ERROR: no input file. Use --input or --generate."); sys.exit(1)

    file_bytes  = os.path.getsize(inp)
    file_mb     = file_bytes / (1024**2)
    n_records   = file_bytes // RECORD_SIZE
    np_list     = [int(x) for x in args.np_list.split(',')]

    print(f"\n{'='*56}")
    print(f"  Distributed Sort Benchmark  (RAM/node: {args.mem} MB)")
    print(f"{'='*56}")
    print(f"  Input  : {inp} ({format_size(file_bytes)}, {n_records:,} records)")
    print(f"  Nodes  : {np_list}")
    print(f"{'='*56}\n")

    results = []
    baseline = None

    for np in np_list:
        print(f"  np={np} ...", end='', flush=True)
        r = run_sort(args.binary, inp, args.output, np, args.mem, args.verify)
        if not r['success']:
            print(f"  FAILED\n  {r['stderr'][-200:]}")
            results.append({**r, 'speedup': None, 'throughput': None})
            continue
        tput    = file_mb / r['elapsed']
        speedup = (baseline / r['elapsed']) if baseline else 1.0
        if baseline is None: baseline = r['elapsed']
        print(f"  {r['elapsed']:.2f}s   {tput:.1f} MB/s   speedup {speedup:.2f}×")
        results.append({**r, 'throughput': tput, 'speedup': speedup})

    print(f"\n{'='*56}")
    print(f"  {'np':>4} | {'Time(s)':>8} | {'MB/s':>7} | {'Speedup':>8}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
    for r in results:
        if r['success']:
            print(f"  {r['np']:>4} | {r['elapsed']:>8.2f} | "
                  f"{r.get('throughput',0):>7.1f} | {r.get('speedup',0):>7.2f}×")
        else:
            print(f"  {r['np']:>4} | {'FAILED':>8}")
    print(f"{'='*56}\n")

    # Parallel efficiency
    good = [r for r in results if r['success'] and r.get('speedup')]
    if len(good) >= 2:
        base_np = good[0]['np']
        print("  Parallel efficiency:")
        for r in good:
            eff = r['speedup'] / r['np'] * base_np * 100
            bar = '█' * max(1, int(eff/5))
            print(f"    np={r['np']:>2}: {eff:5.1f}%  {bar}")

    if args.json:
        with open(args.json, 'w') as jf:
            json.dump(results, jf, indent=2, default=str)
        print(f"\n  Results → {args.json}")

if __name__ == '__main__':
    main()
