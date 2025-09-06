#!/usr/bin/env python3

import argparse
import time
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys
import torch
from collections import defaultdict as ddict

sys.path.insert(0, str(Path(__file__).parent.parent))

from client import PendingRequest
from example_model import NnInferenceClient


class LocalEvaluator:
    def __init__(self, requests_file: str):
        self.requests_df = pd.read_parquet(requests_file)
        symbol_idxs = [int(sym[-3:]) for sym in self.requests_df["symbol"].unique()]
        self.num_symbols = max(symbol_idxs) + 1

    def evaluate_model(self, process_batch_fn, batch_size) -> dict[str, float]:
        all_requests = []
        for idx, row in self.requests_df.iterrows():
            feature_cols = [
                x for x in self.requests_df.columns if x.startswith("feature")
            ]
            features = [float(row[col]) for col in feature_cols]
            req = PendingRequest(
                unique_id=idx,
                symbol=row["symbol"],
                features=features,
                received_time=time.time(),
            )
            all_requests.append(req)

        predictions = {}
        request_latencies = {}

        for i in range(0, len(all_requests), batch_size):
            batch = all_requests[i : i + batch_size]

            batch_by_symbol = {}
            for req in batch:
                if req.symbol not in batch_by_symbol:
                    batch_by_symbol[req.symbol] = []
                batch_by_symbol[req.symbol].append(req)

            start_time = time.perf_counter()
            responses = process_batch_fn(batch_by_symbol)
            end_time = time.perf_counter()

            batch_time_ms = (end_time - start_time) * 1000

            for unique_id, pred in zip(responses.unique_ids, responses.predictions):
                predictions[unique_id] = pred
                request_latencies[unique_id] = batch_time_ms

        return self._calculate_metrics(predictions, request_latencies)

    def _calculate_metrics(self, predictions: dict, request_latencies: dict) -> dict:
        latencies = []
        tower_accuracies = ddict(list)

        target_cols = [x for x in self.requests_df.columns if x.startswith("target")]

        for idx, row in self.requests_df.iterrows():
            unique_id = idx
            targets = [float(row[col]) for col in target_cols]

            if unique_id not in predictions:
                continue

            all_tower_predictions = predictions[unique_id]
            for tower_idx, (tower_pred, tower_target) in enumerate(
                zip(all_tower_predictions, targets, strict=True)
            ):
                tower_accuracies[tower_idx].append(abs(tower_pred - tower_target))

            latency_ms = request_latencies[unique_id]
            latencies.append(latency_ms)

        response_rate = len(predictions) / len(self.requests_df)
        avg_latency = np.mean(latencies) if latencies else 0

        stats_dict = {
            "total_requests": len(self.requests_df),
            "total_responses": len(predictions),
            "response_rate": response_rate,
            "avg_latency_ms": avg_latency,
        }

        for tower_idx, abs_errors in tower_accuracies.items():
            max_error = max(abs_errors)
            mean_error = np.mean(abs_errors)
            stats_dict[f"tower_{tower_idx}"] = (max_error, mean_error)

        return stats_dict

    def print_report(self, metrics: Dict[str, float]):
        print(f"\nTotal requests: {metrics['total_requests']:,}")
        print(f"Total responses: {metrics['total_responses']:,}")
        print(f"Response rate: {metrics['response_rate']:.2%}")
        print(f"Average latency: {metrics['avg_latency_ms']:.1f} ms")

        print("Tower Accuracies:")  # We have 4 towers
        for tower_idx, tower_name in enumerate(["XLSTM", "Mamba2", "RetNet", "Hawk"]):
            max_error, mean_error = metrics[f"tower_{tower_idx}"]
            print(f"{tower_name}: {max_error = :.4f}, {mean_error = :.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-parquet-file", type=str, default="tiny.parquet")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    evaluator = LocalEvaluator(args.requests_parquet_file)
    client = NnInferenceClient(num_symbols=evaluator.num_symbols, token=args.token)

    metrics = evaluator.evaluate_model(client.process_batch, batch_size=args.batch_size)
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
