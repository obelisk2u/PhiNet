import argparse
import os
import numpy as np
import pandas as pd
import torch


def load_and_clean(
    csv_path: str,
    time_col: str = "Call Date Time",
    address_col: str = "Address",
    city_col: str | None = "City",
    city_filter: str | None = "San Francisco",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    cols = [time_col, address_col] + ([city_col] if city_col is not None and city_col in df.columns else [])
    df = df[cols]
    df = df.dropna(subset=[time_col, address_col])

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def build_sequences_from_addresses(
    df: pd.DataFrame,
    time_col: str = "Call Date Time",
    address_col: str = "Address",
    top_k_addresses: int = 100,
    min_events_per_addr: int = 50,
    max_events_per_seq: int = 500,
) -> dict:
    #take the top k addresses by event count and build per-address sequences
    addr_counts = df[address_col].value_counts()
    addr_counts = addr_counts[addr_counts >= min_events_per_addr]
    top_addrs = addr_counts.head(top_k_addresses).index.tolist()
    print(
        f"Selected {len(top_addrs)} addresses "
        f"(top_k={top_k_addresses}, min_events={min_events_per_addr}, "
        f"max_events_per_seq={max_events_per_seq})."
    )

    sequences = {}

    for addr in top_addrs:
        sub = df[df[address_col] == addr].sort_values(time_col)
        times = sub[time_col].values   

        if len(times) == 0:
            continue

        #truncate
        if len(times) > max_events_per_seq:
            times = times[:max_events_per_seq]

        t0 = times[0]
        times_sec = (times - t0) / np.timedelta64(1, "s")
        sequences[addr] = times_sec.astype(np.float32)

    return sequences


def times_to_interarrivals(
    sequences: dict,
    train_frac: float = 0.8,
    min_train_events: int = 20,
    min_test_events: int = 10,
):
    #convert event times to inter-arrival times and split
    train_seqs = []
    test_seqs = []

    for addr, times in sequences.items():
        times = np.asarray(times)
        n = len(times)
        if n < (min_train_events + min_test_events):
            continue

        #inter-arrivals
        deltas = np.empty_like(times)
        deltas[0] = times[0]
        deltas[1:] = times[1:] - times[:-1]

        deltas = np.clip(deltas, a_min=1e-8, a_max=None)

        #split
        split_idx = int(np.floor(train_frac * n))
        if split_idx < min_train_events or (n - split_idx) < min_test_events:
            continue

        train_deltas = deltas[:split_idx]
        test_deltas = deltas[split_idx:]

        train_seqs.append(torch.from_numpy(train_deltas.astype(np.float32)))
        test_seqs.append(torch.from_numpy(test_deltas.astype(np.float32)))

    print(f"Built {len(train_seqs)} train sequences and {len(test_seqs)} test sequences.")
    return train_seqs, test_seqs


def main():
    parser = argparse.ArgumentParser(description="Preprocess SF 911 calls into TPP sequences.")
    parser.add_argument("--csv", type=str, required=True, help="Path to raw SF calls CSV file.")
    parser.add_argument(
        "--out",
        type=str,
        default="sf_tpp_sequences.pt",
        help="Output path for saved PyTorch sequences.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top K addresses by event count.")
    parser.add_argument("--min-events", type=int, default=50, help="Minimum events per address.")
    parser.add_argument("--max-events-per-seq", type=int, default=500,
                        help="Maximum number of events to keep per address sequence.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction per sequence.")
    parser.add_argument("--min-train-events", type=int, default=20, help="Minimum train events per sequence.")
    parser.add_argument("--min-test-events", type=int, default=10, help="Minimum test events per sequence.")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    print(f"Loading data from {args.csv} ...")
    df = load_and_clean(csv_path=args.csv)

    print("Building per-address sequences of event times ...")
    seq_times = build_sequences_from_addresses(
        df,
        top_k_addresses=args.top_k,
        min_events_per_addr=args.min_events,
        max_events_per_seq=args.max_events_per_seq,
    )

    print("Converting times to inter-arrival sequences and splitting train/test ...")
    train_seqs, test_seqs = times_to_interarrivals(
        seq_times,
        train_frac=args.train_frac,
        min_train_events=args.min_train_events,
        min_test_events=args.min_test_events,
    )

    out_obj = {
        "train_seqs": train_seqs,
        "test_seqs": test_seqs,
        "meta": {
            "top_k": args.top_k,
            "min_events_per_addr": args.min_events,
            "max_events_per_seq": args.max_events_per_seq,
            "train_frac": args.train_frac,
            "min_train_events": args.min_train_events,
            "min_test_events": args.min_test_events,
        },
    }

    torch.save(out_obj, args.out)
    print(f"Saved sequences to {args.out}")
    print(f"Example train seq length(s): {[len(s) for s in train_seqs[:5]]}")
    print(f"Example test  seq length(s): {[len(s) for s in test_seqs[:5]]}")


if __name__ == "__main__":
    main()