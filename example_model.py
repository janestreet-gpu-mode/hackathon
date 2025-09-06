#!/usr/bin/env python3
"""
Example model implementation for GPU inference game.
Shows how to implement the BaseInferenceClient.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import argparse
from typing import Dict, List
import torch

from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
    ):
        super().__init__(num_symbols, server_host, server_port)

        self.device = device or get_default_device()

        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        self.model = MultiTowerModel(config).to(self.device)

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        self.states = {
            f"SYM_{num:03d}": self.model.init_state(1, self.device)
            for num in range(self.num_symbols)
        }

        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        unique_ids, preds = [], []

        start = time.time()

        for symbol, symbol_requests in requests_by_symbol.items():
            state = self.states[symbol]
            for req in symbol_requests:
                features = torch.tensor([req.features], device=self.device)

                with torch.inference_mode():
                    pred, state = self.model(features, state)

                unique_ids.append(req.unique_id)
                preds.append(pred.cpu().squeeze(0).numpy().astype(float).tolist())

            self.states[symbol] = state

        end = time.time()
        elapsed = end - start

        # May not be a bad idea to print less often!
        print(f"{len(preds) = }, {elapsed = }, {elapsed / len(preds) = }")

        return InferenceResponse(
            unique_ids=unique_ids, predictions=preds, client_timestamp=time.time()
        )


def main():
    parser = argparse.ArgumentParser(description="Example inference client")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols in the tradeable universe",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token to download the model (for testing before the hackathon)",
    )

    args = parser.parse_args()
    client = NnInferenceClient(
        num_symbols=args.num_symbols,
        server_host=args.host,
        server_port=args.port,
        token=args.token,
    )

    client.run()


if __name__ == "__main__":
    main()
