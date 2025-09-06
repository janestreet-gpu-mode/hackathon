"""
Protocol definitions for the GPU inference game.
Shared between server and client.
"""

import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from enum import Enum


class MessageType(Enum):
    REGISTER = "register"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    SCORE_UPDATE = "score_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    STATS_UPDATE = "stats_update"


@dataclass
class RegisterMessage:
    message_type: str = MessageType.REGISTER.value


@dataclass
class InferenceRequest:
    unique_ids: list[int]
    symbols: list[str]
    features: list[list[float]]
    timestamp: float  # When sent by server
    message_type: str = MessageType.INFERENCE_REQUEST.value


@dataclass
class InferenceResponse:
    unique_ids: list[int]
    predictions: list[list[float]]
    client_timestamp: float  # When client processed
    message_type: str = MessageType.INFERENCE_RESPONSE.value


@dataclass
class ScoreUpdate:
    unique_ids: list[int]
    trade_pnls: list[float]
    accuracies: list[float]
    latencies_ms: list[float]
    message_type: str = MessageType.SCORE_UPDATE.value


@dataclass
class Heartbeat:
    timestamp: float
    message_type: str = MessageType.HEARTBEAT.value


@dataclass
class ErrorMessage:
    error: str
    details: Optional[str] = None
    message_type: str = MessageType.ERROR.value


class ProtocolHandler:
    """Handles encoding/decoding of protocol messages."""

    MESSAGE_CLASSES = {
        MessageType.REGISTER.value: RegisterMessage,
        MessageType.INFERENCE_REQUEST.value: InferenceRequest,
        MessageType.INFERENCE_RESPONSE.value: InferenceResponse,
        MessageType.SCORE_UPDATE.value: ScoreUpdate,
        MessageType.HEARTBEAT.value: Heartbeat,
        MessageType.ERROR.value: ErrorMessage,
    }

    @staticmethod
    def encode(message: Any) -> bytes:
        """Encode a message object to JSON bytes with newline."""
        data = asdict(message)
        return (json.dumps(data) + "\n").encode("utf-8")

    @staticmethod
    def decode(data: bytes) -> Any:
        """Decode JSON bytes to appropriate message object."""
        try:
            msg_dict = json.loads(data.decode("utf-8").strip())
            msg_type = msg_dict.get("message_type")

            if msg_type not in ProtocolHandler.MESSAGE_CLASSES:
                raise ValueError(f"Unknown message type: {msg_type}")

            cls = ProtocolHandler.MESSAGE_CLASSES[msg_type]
            # Remove message_type from dict before creating object
            msg_dict.pop("message_type", None)
            obj = cls(**msg_dict)
            obj.message_type = msg_type
            return obj
        except Exception as e:
            return ErrorMessage(error=str(e))


class SocketReader:
    """Buffered socket reader for handling newline-delimited messages."""

    def __init__(self, sock: socket.socket, buffer_size: int = 4096):
        self.sock = sock
        self.buffer = b""
        self.buffer_size = buffer_size

    def read_message(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Read one complete message from socket."""
        if timeout:
            self.sock.settimeout(timeout)

        while b"\n" not in self.buffer:
            try:
                data = self.sock.recv(self.buffer_size)
                if not data:
                    return None  # Connection closed
                self.buffer += data
            except socket.timeout:
                return None
            except Exception:
                return None

        # Extract one complete message
        line, self.buffer = self.buffer.split(b"\n", 1)
        if line:
            return ProtocolHandler.decode(line)
        return None

    def read_all_available(self) -> List[Any]:
        """Read all currently available messages without blocking."""
        messages = []
        self.sock.setblocking(False)

        try:
            # First, read all available data
            while True:
                try:
                    data = self.sock.recv(self.buffer_size)
                    if not data:
                        break
                    self.buffer += data
                except socket.error:
                    break

            # Then parse all complete messages
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line:
                    msg = ProtocolHandler.decode(line)
                    if msg:
                        messages.append(msg)
        finally:
            self.sock.setblocking(True)

        return messages


class SocketWriter:
    """Socket writer for sending protocol messages."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send_message(self, message: Any) -> bool:
        """Send a message over the socket."""
        try:
            data = ProtocolHandler.encode(message)
            self.sock.sendall(data)
            return True
        except Exception:
            return False
