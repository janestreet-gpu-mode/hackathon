"""
Client framework for GPU inference game.
Provides base class for participants to implement.
"""

import socket
import threading
import time
import queue
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from protocol import (
    ProtocolHandler,
    SocketReader,
    SocketWriter,
    RegisterMessage,
    InferenceRequest,
    InferenceResponse,
    ScoreUpdate,
    Heartbeat,
    ErrorMessage,
)


@dataclass
class PendingRequest:
    """A pending inference request."""

    unique_id: int
    symbol: str
    features: List[float]
    received_time: float

    def age_ms(self) -> float:
        """Get age of request in milliseconds."""
        return (time.time() - self.received_time) * 1000


class BaseInferenceClient(ABC):
    """
    Base class for inference clients.
    Participants should inherit from this and implement process_batch().
    """

    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        max_queue_size: int = 100,
    ):
        """
        Initialize the client.

        Args:
            server_host: Server hostname
            server_port: Server TCP port
            max_queue_size: Maximum pending requests per symbol
        """

        self.num_symbols = num_symbols
        self.server_host = server_host
        self.server_port = server_port
        self.max_queue_size = max_queue_size

        # Connection
        self.socket = None
        self.reader = None
        self.writer = None

        # Request queues by symbol
        self.request_queues: Dict[str, queue.Queue] = defaultdict(
            lambda: queue.Queue(maxsize=max_queue_size)
        )
        self.queue_lock = threading.RLock()

        # Threading
        self.running = False
        self.receive_thread = None
        self.process_thread = None

        # Response tracking
        self.response_queue = queue.Queue()

    @abstractmethod
    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        pass

    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))

            self.reader = SocketReader(self.socket)
            self.writer = SocketWriter(self.socket)

            # Send registration
            reg_msg = RegisterMessage()
            if not self.writer.send_message(reg_msg):
                print(f"Failed to send registration")
                return False

            print(f"Connected to {self.server_host}:{self.server_port}")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

    def _receive_loop(self):
        """Background thread to receive messages from server."""

        last_heartbeat_send = 0
        assert self.reader
        assert self.writer

        while self.running:
            try:
                messages = self.reader.read_all_available()

                for msg in messages:
                    if isinstance(msg, InferenceRequest):
                        self._handle_request(msg)
                    elif isinstance(msg, ScoreUpdate):
                        self._handle_score(msg)
                    elif isinstance(msg, ErrorMessage):
                        print(f"Server error: {msg.error}")

                time_seconds = int(time.time())
                if time_seconds % 5 == 0 and (time_seconds - last_heartbeat_send) >= 5:
                    self.writer.send_message(Heartbeat(timestamp=time.time()))
                    last_heartbeat_send = time_seconds

                time.sleep(0.001)

            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                    break

    def _handle_request(self, request: InferenceRequest):
        """Handle incoming inference request."""

        for unique_id, symbol, features in zip(
            request.unique_ids, request.symbols, request.features
        ):
            pending = PendingRequest(
                unique_id=unique_id,
                symbol=symbol,
                features=features,
                received_time=time.time(),
            )

            with self.queue_lock:
                try:
                    self.request_queues[symbol].put_nowait(pending)
                except queue.Full:
                    print(f"Queue full for symbol {symbol}, dropping request")

    def _handle_score(self, score: ScoreUpdate):
        """Handle score update from server."""
        pass

    def _process_loop(self):
        """Background thread to process inference requests."""
        while self.running:
            try:
                # Gather current requests by symbol
                requests_by_symbol = self._gather_requests()

                if requests_by_symbol:
                    # Call user's implementation
                    response = self.process_batch(requests_by_symbol)
                    if self.writer and not self.writer.send_message(response):
                        print(f"Failed to send response for {response.unique_ids = }")
                else:
                    # No requests, small sleep
                    time.sleep(0.001)

            except Exception as e:
                print(f"Process error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)

    def _gather_requests(self) -> Dict[str, List[PendingRequest]]:
        """Gather all pending requests by symbol."""
        requests_by_symbol = {}

        with self.queue_lock:
            for symbol, q in self.request_queues.items():
                requests = []

                while not q.empty():
                    try:
                        req = q.get_nowait()
                        requests.append(req)
                    except queue.Empty:
                        break

                if len(requests) > 0:
                    requests_by_symbol[symbol] = requests

        return requests_by_symbol

    def run(self):
        """Main client run loop."""
        if not self.connect():
            return

        self.running = True

        # Start background threads
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        print(f"Client running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the client."""
        self.running = False

        # Wait for threads
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        if self.process_thread:
            self.process_thread.join(timeout=1)

        self.disconnect()
