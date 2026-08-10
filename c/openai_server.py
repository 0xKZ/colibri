#!/usr/bin/env python3
"""Dependency-free OpenAI-compatible HTTP gateway for the colibri engine."""

import argparse
import codecs
import collections
import contextlib
import hashlib
import json
import math
import mimetypes
import os
import select
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


HERE = Path(__file__).resolve().parent


def default_engine():
    """The engine next to this file. Since #391 it is built as `colibri`; `glm` stays as a
    fallback so an old tree (or an old hand-built binary) still starts. Reported by
    @RDouglasSharp in #488: the default still said `glm`, so `python3 openai_server.py`
    on a clean checkout looked for a file the build no longer produces."""
    for name in ("colibri", "colibri.exe", "glm", "glm.exe"):
        candidate = HERE / name
        if candidate.exists():
            return candidate
    return HERE / "colibri"
END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"
MAX_BODY = 4 << 20
PROFILE_TURNS = 120           # rolling window of per-turn PROF snapshots kept for /profile
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://tauri.localhost",
    "tauri://localhost",
)


class APIError(Exception):
    def __init__(self, status, message, param=None, code=None, error_type="invalid_request_error",
                 headers=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param
        self.code = code
        self.error_type = error_type
        self.headers = headers or {}


class ClientCancelled(Exception):
    pass


def error_object(error):
    return {"error": {"message": error.message, "type": error.error_type,
                      "param": error.param, "code": error.code}}


def _engine_error(fields, message):
    """Turn an engine ERROR frame into the right exception type.

    CONTEXT_EXCEEDED is a client mistake, not a server fault: the prompt is longer than the
    engine's context. Report it the way every OpenAI-compatible server does, so clients that
    know how to compact a conversation actually get the chance to (previously the engine
    silently truncated the prompt instead, which is #401)."""
    if fields and fields[0] == "CONTEXT_EXCEEDED":
        limit = fields[2] if len(fields) > 2 else "the context"
        used = fields[1] if len(fields) > 1 else "?"
        return APIError(400,
                        f"This model's maximum context length is {limit} tokens, however your "
                        f"messages resulted in at least {used} tokens. Please shorten the "
                        f"conversation, or restart the server with a larger CTX.",
                        "messages", "context_length_exceeded")
    return RuntimeError(message)


class GenerationScheduler:
    """Bounded FIFO admission for the engine's independent KV contexts."""

    def __init__(self, max_queue=8, queue_timeout=300, capacity=1):
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.capacity = capacity
        self.free_slots = set(range(capacity))
        self.condition = threading.Condition()
        self.queue = collections.deque()
        self.active = 0
        self.closed = False
        self.admitted = 0
        self.completed = 0
        self.rejected = 0
        self.timed_out = 0
        self.cancelled = 0

    @contextlib.contextmanager
    def admit(self, cancelled=None, slot=None):
        ticket = object()
        entry = (ticket, slot)          # (#B2) remember each waiter's target slot for fair, per-slot admission
        queued_at = time.monotonic()
        with self.condition:
            if self.closed:
                raise APIError(503, "The inference scheduler is shutting down.", None,
                               "scheduler_closed", "server_error")
            if (self.active >= self.capacity or self.queue) and len(self.queue) >= self.max_queue:
                self.rejected += 1
                raise APIError(429, "The inference queue is full.", None, "queue_full",
                               "rate_limit_error", {"Retry-After": "1"})
            self.queue.append(entry)
            deadline = queued_at + self.queue_timeout
            while True:
                if self.closed:
                    self.queue.remove(entry)
                    self.condition.notify_all()
                    raise APIError(503, "The inference scheduler is shutting down.", None,
                                   "scheduler_closed", "server_error")
                available = min(self.free_slots) if slot is None and self.free_slots else slot
                # (#B2) Admit as soon as our target slot is free AND no strictly-earlier
                # waiter also wants it (an earlier waiter "wants" it if it is any-slot or
                # pinned to the same slot). This replaces the old strict FIFO-head rule,
                # which let a head pinned to a busy slot block every request behind it —
                # even ones targeting a currently-free slot (head-of-line blocking).
                # ponytail: O(queue) scan per wakeup — negligible at the default max_queue;
                # switch to per-slot wait sets if max_queue is ever raised to thousands.
                can_admit = available in self.free_slots
                if can_admit:
                    for t2, s2 in self.queue:
                        if t2 is ticket:
                            break
                        if s2 is None or s2 == available:
                            can_admit = False
                            break
                if can_admit:
                    break
                if cancelled and cancelled():
                    self.queue.remove(entry)
                    self.cancelled += 1
                    self.condition.notify_all()
                    raise ClientCancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.queue.remove(entry)
                    self.timed_out += 1
                    self.condition.notify_all()
                    raise APIError(429, "Timed out waiting for the inference engine.", None,
                                   "queue_timeout", "rate_limit_error", {"Retry-After": "1"})
                self.condition.wait(min(remaining, 0.25))
            self.queue.remove(entry)
            self.free_slots.remove(available)
            self.active += 1
            self.admitted += 1
            wait_seconds = time.monotonic() - queued_at
        try:
            yield wait_seconds, available
        finally:
            with self.condition:
                self.active -= 1
                self.free_slots.add(available)
                self.completed += 1
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "queued": len(self.queue),
                    "capacity": self.capacity,
                    "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout,
                    "admitted": self.admitted, "completed": self.completed,
                    "rejected": self.rejected, "timed_out": self.timed_out,
                    "cancelled": self.cancelled}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def content_text(content, param):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of text parts.", param)
    parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") not in ("text", "input_text"):
            raise APIError(400, "Colibri currently supports text message content only.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(part.get("text"), str):
            raise APIError(400, "Text content parts require a string `text` field.",
                           f"{param}.{index}.text")
        parts.append(part["text"])
    return "".join(parts)


# ---- GLM-5.2 tool calling -----------------------------------------------------------------
# The model expresses tool calls as ordinary text (from chat_template.jinja):
#    <tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>
# and tool results come back as <|observation|><tool_response>{content}</tool_response>.
# We render those markers into the prompt and parse them back into OpenAI `tool_calls`.
import re

BOX_START, BOX_END = "<tool_call>", "</tool_call>"
TR_OPEN,  TR_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_BOX_RE  = re.compile(re.escape(BOX_START) + r"(.*?)" + re.escape(BOX_END), re.DOTALL)
_ARG_RE  = re.compile(r"<arg_key>([^<]*)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL)
_NAME_RE = re.compile(r"\s*([A-Za-z0-9_.\-]+)")
_TAG_RE  = re.compile(r"</?arg_key>|</?arg_value>")
# A closing tag the model started but never finished ("</tool_cal", "</tool"), at end of reply.
_PARTIAL_END_RE = re.compile(r"<(?:/(?:t(?:o(?:o(?:l(?:_(?:c(?:a(?:l)?)?)?)?)?)?)?)?)?\Z")

# De-mangler: opt-in recovery for heavily-quantized models that drop the
# <arg_key>K</arg_key><arg_value> structure. Default OFF (never rewrites well-formed output).
_SALVAGE = os.environ.get("COLI_TOOL_SALVAGE", "0") == "1"


def _tool_param_order(tools):
    """name -> ordered param names (required first) from the request schema, for de-mangling."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        params = ((fn.get("parameters") or {}).get("properties") or {})
        required = list((fn.get("parameters") or {}).get("required") or [])
        out[name] = required + [p for p in params if p not in required]
    return out


def _tool_param_types(tools):
    """name -> {param: declared JSON-schema type}. The model emits every argument as text;
    without the schema a string-typed value that happens to look numeric ("12345" for an
    order id, an SKU, a phone number) would be json.loads()'d into an int and the tool would
    receive the wrong type."""
    out = {}
    for tool in (tools or []):
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types = {}
        for key, spec in props.items():
            if isinstance(spec, dict):
                t = spec.get("type")
                if isinstance(t, list):          # {"type": ["string", "null"]}
                    t = next((x for x in t if x != "null"), None)
                types[key] = t
        out[name] = types
    return out


def _coerce_arg(value, declared):
    """Decode a raw <arg_value> according to the declared schema type.

    A string-typed parameter is kept verbatim -- never parsed as JSON. Everything else keeps
    the previous permissive behaviour (parse if it parses, otherwise leave as text)."""
    if declared == "string":
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    if declared in ("integer", "number") and isinstance(parsed, bool):
        return value                              # `true` is not a number
    if declared and declared not in ("integer", "number", "boolean", "object", "array"):
        return value
    return parsed


def _unclosed_tail(reply, tools):
    """Body of a trailing <tool_call> that was never closed, or None.

    Only returned when the recovery is unambiguous, so ordinary prose that merely mentions
    "<tool_call>" can never be turned into a call. Both conditions must hold:
      * the last BOX_START is not followed by a BOX_END (a closed box is the strict parser's job);
      * the tail carries a complete <arg_key>..</arg_value> pair, OR it is exactly the name of a
        tool the client declared (the zero-argument case).
    """
    start = reply.rfind(BOX_START)
    if start < 0 or BOX_END in reply[start:]:
        return None
    inner = _PARTIAL_END_RE.sub("", reply[start + len(BOX_START):])
    if _ARG_RE.search(inner):
        return inner
    declared = {(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                for t in (tools or []) if isinstance(t, dict)}
    return inner if inner.strip() in declared else None


def parse_tool_calls(reply, tools=None):
    """Return (content, tool_calls). Strict GLM parse; optional de-mangler (COLI_TOOL_SALVAGE=1)
    rescues malformed int4 output by mapping a lone payload onto the tool's primary parameter."""
    param_order = _tool_param_order(tools)
    param_types = _tool_param_types(tools)
    calls, salvaged = [], []
    # #401: a box the model opened but never closed -- it ran out of budget, or the closing tag
    # came out mangled ("</tool_cal"). The call itself is often perfectly well-formed, but the
    # strict regex needs BOTH tags, so the client used to get *zero* tool_calls. Recover the tail,
    # but only when it is unambiguous (see _unclosed_tail) so prose can never fabricate a call.
    boxes = [m.group(1) for m in _BOX_RE.finditer(reply)]
    tail = _unclosed_tail(reply, tools)
    if tail is not None:
        boxes.append(tail)
    for inner in boxes:
        name_match = _NAME_RE.match(inner)
        name = name_match.group(1) if name_match else inner.strip()
        args = {}
        types = param_types.get(name, {})
        for arg in _ARG_RE.finditer(inner):
            key, value = arg.group(1), arg.group(2)
            args[key] = _coerce_arg(value, types.get(key))
        if not args and _SALVAGE:
            rest = inner[name_match.end():] if name_match else ""
            payload = _TAG_RE.sub("", rest).strip()
            if payload.startswith("(") and payload.endswith(")"):
                payload = payload[1:-1].strip()
            if payload:
                key = (param_order.get(name) or ["input"])[0]
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                args = {key: payload}
                salvaged.append(name)
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if tools and not calls and re.search(r"</?tool_call>|</?arg_key>|</?arg_value>", reply):
        # Diagnosi per la #401: il client ha dichiarato i tools e il modello ha PROVATO la
        # sintassi, ma il parse rigoroso non ha agganciato nulla (tipico output int4 storpiato).
        # EN: #401 field diagnosis: tools were declared and the model attempted the syntax,
        # EN: but the strict parse matched nothing (typically quantization-mangled output).
        sys.stderr.write("[api] tools declared and tool-call markers present, but no call "
                         "parsed -- output may be quantization-mangled; try COLI_TOOL_SALVAGE=1\n")
        sys.stderr.flush()
    text = _BOX_RE.sub("", reply)
    if tail is not None:                       # drop the recovered tail from the visible content
        text = text[:text.rindex(BOX_START)]
    if ARCH == "inkling":
        text = strip_inkling_markers(text)   # thinking is reasoning, not answer
    if THINK_CLOSE in text:
        text = text.split(THINK_CLOSE, 1)[1]
    text = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    if calls:
        dm, rec = len(salvaged), (1 if tail is not None else 0)
        sys.stderr.write("[api] tool-calls: %d total, %d strict, %d unclosed-recovered, "
                         "%d de-mangled [%s]%s\n"
                         % (len(calls), max(0, len(calls) - dm - rec), rec, dm,
                            "CLEAN" if dm == 0 and rec == 0 else "RECOVERED",
                            (" -> " + ", ".join(salvaged)) if dm else ""))
        sys.stderr.flush()
    return text.strip(), calls


ARCH = "glm"   # set in main(): glm | inkling | kimi | deepseek_v4

# ---- DeepSeek V4 tool calling -----------------------------------------------------------
# DSML format:
#   Tool calls: <｜DSML｜tool_calls> ... <｜DSML｜invoke name="X"> ... </｜DSML｜tool_calls>
#   Parameters: <｜DSML｜parameter name="K" string="true|false">V</｜DSML｜parameter>
#   Tool results: <tool_result>...</tool_result> (merged into user messages)
DSML = "｜DSML｜"
DSML_TOOL_CALLS_OPEN = f"<{DSML}tool_calls>"
DSML_TOOL_CALLS_CLOSE = f"</{DSML}tool_calls>"
DSML_INVOKE_OPEN = f"<{DSML}invoke"
DSML_INVOKE_CLOSE = f"</{DSML}invoke>"
DSML_PARAM_TAG = f"<{DSML}parameter"
DSML_PARAM_CLOSE = f"</{DSML}parameter>"
DSML_PARAM_RE = re.compile(
    r'<' + re.escape(DSML) + r'parameter\s+name="([^"]+)"\s+string="(true|false)">([^<]*)</' + re.escape(DSML) + r'parameter>',
    re.DOTALL
)
DSML_INVOKE_RE = re.compile(
    r'<' + re.escape(DSML) + r'invoke\s+name="([^"]+)">',
    re.DOTALL
)
DSML_TOOL_CALLS_BLOCK_RE = re.compile(
    re.escape(DSML_TOOL_CALLS_OPEN) + r"(.*?)" + re.escape(DSML_TOOL_CALLS_CLOSE),
    re.DOTALL
)
DSML_TOOL_RESULT_OPEN = "<tool_result>"
DSML_TOOL_RESULT_CLOSE = "</tool_result>"


def parse_tool_calls_v4(reply, tools=None):
    """Return (content, tool_calls) for DeepSeek V4 DSML format."""
    calls = []
    for block_match in DSML_TOOL_CALLS_BLOCK_RE.finditer(reply):
        block_content = block_match.group(1)
        for invoke_match in DSML_INVOKE_RE.finditer(block_content):
            tool_name = invoke_match.group(1)
            # Find the scope of this invoke block (up to </invoke>)
            invoke_start = invoke_match.end()
            rest = block_content[invoke_start:]
            invoke_end = rest.find(DSML_INVOKE_CLOSE)
            if invoke_end < 0:
                invoke_end = len(rest)
            invoke_body = rest[:invoke_end]
            # Parse parameters into a proper dict, then serialize
            parsed_args = {}
            for param_match in DSML_PARAM_RE.finditer(invoke_body):
                param_name, is_string, param_value = param_match.group(1), param_match.group(2), param_match.group(3)
                if is_string == "true":
                    parsed_args[param_name] = param_value
                else:
                    try:
                        parsed_args[param_name] = json.loads(param_value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args[param_name] = param_value
            calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(parsed_args, ensure_ascii=False)
                }
            })
    # Strip tool call blocks from content
    text = DSML_TOOL_CALLS_BLOCK_RE.sub("", reply)
    if calls:
        sys.stderr.write(f"[api] v4 tool-calls: {len(calls)} parsed\n")
        sys.stderr.flush()
    return text.strip(), calls


# ---- Inkling ---------------------------------------------------------------------------

INK_THINK, INK_TEXT = "<|content_thinking|>", "<|content_text|>"


class InklingStreamSplit:
    """Strips Inkling's content markers from the visible stream and withholds
    <|content_thinking|> sections from `content` (they are reasoning, not
    answer). Buffers partial markers across chunk boundaries so a marker split
    between two DATA frames never leaks."""

    def __init__(self, on_content, on_reasoning=None):
        self.on_content = on_content
        self.on_reasoning = on_reasoning
        self.mode = "content"
        self.buf = ""

    def feed(self, piece):
        self.buf += piece
        while True:
            hits = [(i, m) for i, m in ((self.buf.find(INK_THINK), INK_THINK),
                                        (self.buf.find(INK_TEXT), INK_TEXT)) if i >= 0]
            if not hits:
                hold = self._tail_hold()
                out = self.buf[:len(self.buf) - hold] if hold else self.buf
                self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                self._emit(out)
                return
            i, m = min(hits)
            self._emit(self.buf[:i])
            self.mode = "reasoning" if m == INK_THINK else "content"
            self.buf = self.buf[i + len(m):]

    def _tail_hold(self):
        for k in range(min(len(self.buf), 24), 0, -1):
            if INK_THINK.startswith(self.buf[-k:]) or INK_TEXT.startswith(self.buf[-k:]):
                return k
        return 0

    def _emit(self, text):
        if not text:
            return
        text = _INK_MARKER.sub("", text)
        if not text:
            return
        if self.mode == "content":
            self.on_content(text)
        elif self.on_reasoning:
            self.on_reasoning(text)

    def close(self):
        self._emit(self.buf)
        self.buf = ""


import re as _re
_INK_MARKER = _re.compile(r"<\|(?:content_\w+|end_message|message_\w+|audio_end|unused_\d+)\|>")

def strip_inkling_markers(text):
    """Remove <|content_thinking|>…<|content_text|> sections, then any stray
    control markers (end_message, role/content tokens) the model emits."""
    while INK_THINK in text:
        pre, _, rest = text.partition(INK_THINK)
        _, _, after = rest.partition(INK_TEXT)
        text = pre + after
    return _INK_MARKER.sub("", text)


def split_inkling(text):
    """Split raw Inkling output into (content, reasoning). Thinking blocks
    (<|content_thinking|>…<|content_text|>) become reasoning — including an
    UNTERMINATED trailing block (budget ran out mid-thought), which partitions
    to everything after the opener — so a think-only generation surfaces its
    reasoning instead of collapsing to an empty answer."""
    reasoning = []
    while INK_THINK in text:
        pre, _, rest = text.partition(INK_THINK)
        think, _, after = rest.partition(INK_TEXT)
        reasoning.append(think)
        text = pre + after
    return _INK_MARKER.sub("", text), _INK_MARKER.sub("", "".join(reasoning))


# ---- Inkling DMel audio input ------------------------------------------------------------
# Inkling takes audio as discretized log-mel frames ("DMel"): 80 slaney mel
# bands per 50 ms hop, quantized to 16 levels in log10 [-7, 2]. One frame = one
# <|audio|> placeholder token; the engine swaps in the frame's embedding at that
# position. The DSP below matches tml-renderers 0.1.0 (via tinkernel-audio,
# which is byte-golden against the official wheel): 100 ms periodic-Hann window
# centered on i*hop with zero edge padding, magnitude-domain mel projection,
# and a turn-level RMS boost for quiet audio (rms < 0.01).

AUDIO_SAMPLE_RATE = 16_000
AUDIO_HOP = 800
AUDIO_WINDOW = 1_600
AUDIO_MEL_BANDS = 80
AUDIO_DMEL_LEVELS = 16
AUDIO_DMEL_MIN, AUDIO_DMEL_MAX = -7.0, 2.0
AUDIO_RMS_FLOOR = 0.01
AUDIO_LOG_FLOOR = 1.0e-10

_MEL_FILTERS = None


def _np():
    try:
        import numpy
    except ImportError:
        raise APIError(400, "Audio input needs numpy on the gateway (pip install numpy).",
                       None, "unsupported_content_type")
    return numpy


def _mel_filters(np):
    """Slaney mel filter bank, [80, 801], normalization 2/(upper-lower)."""
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS
    fft_freqs = np.arange(AUDIO_WINDOW // 2 + 1, dtype=np.float64) * AUDIO_SAMPLE_RATE / AUDIO_WINDOW

    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz >= 1000.0, 15.0 + np.log(np.maximum(hz, 1e-30) / 1000.0) / 0.06875177742094912,
                        hz / 66.66666666666667)

    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        return np.where(mel >= 15.0, 1000.0 * np.exp(0.06875177742094912 * (mel - 15.0)),
                        66.66666666666667 * mel)

    max_mel = hz_to_mel(AUDIO_SAMPLE_RATE / 2.0)
    mel_points = mel_to_hz(np.linspace(0.0, float(max_mel), AUDIO_MEL_BANDS + 2))
    lower, center, upper = mel_points[:-2], mel_points[1:-1], mel_points[2:]
    rising = (fft_freqs[None, :] - lower[:, None]) / (center - lower)[:, None]
    falling = (upper[:, None] - fft_freqs[None, :]) / (upper - center)[:, None]
    weights = np.maximum(np.minimum(rising, falling), 0.0) * (2.0 / (upper - lower))[:, None]
    _MEL_FILTERS = weights.astype(np.float32)
    return _MEL_FILTERS


def dmel_encode(samples):
    """Mono 16 kHz f32 PCM -> u8 DMel bytes, [ceil(n/800), 80] row-major."""
    np = _np()
    samples = np.asarray(samples, dtype=np.float32)
    n = samples.shape[0]
    if n == 0:
        raise APIError(400, "Audio clip is empty.", None, "invalid_value")
    frames = -(-n // AUDIO_HOP)
    half = AUDIO_WINDOW // 2
    padded = np.zeros(half + frames * AUDIO_HOP + half, dtype=np.float32)
    padded[half:half + n] = samples
    idx = (np.arange(frames)[:, None] * AUDIO_HOP) + np.arange(AUDIO_WINDOW)[None, :]
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(AUDIO_WINDOW, dtype=np.float64)
                               / AUDIO_WINDOW)).astype(np.float32)
    windows = padded[idx] * hann[None, :]
    sqmag = np.abs(np.fft.rfft(windows, axis=1)) ** 2                   # [frames, 801]
    rms = math.sqrt(float(np.sum(samples.astype(np.float64) ** 2)) / n)
    scale = AUDIO_RMS_FLOOR / rms if 0.0 < rms < AUDIO_RMS_FLOOR else 1.0
    mag = np.sqrt(np.maximum(sqmag * (scale * scale), AUDIO_LOG_FLOOR)).astype(np.float32)
    energy = mag @ _mel_filters(np).T                                   # [frames, 80]
    logmel = np.log10(np.maximum(energy, AUDIO_LOG_FLOOR))
    norm = np.clip((np.clip(logmel, AUDIO_DMEL_MIN, AUDIO_DMEL_MAX) - AUDIO_DMEL_MIN)
                   / (AUDIO_DMEL_MAX - AUDIO_DMEL_MIN), 0.0, 1.0)
    q = np.clip(np.ceil(norm * (AUDIO_DMEL_LEVELS - 1) - 0.5), 0, AUDIO_DMEL_LEVELS - 1)
    return q.astype(np.uint8).tobytes()


def decode_wav_mono16k(data, param):
    """Minimal RIFF/WAVE reader: PCM16 or float32, any channel count (mixed
    down), sample rate must already be 16 kHz — resampling belongs at the
    capture edge, not in the gateway."""
    import struct as _struct
    np = _np()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise APIError(400, "Audio must be a RIFF/WAVE file.", param, "invalid_value")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], _struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = _struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise APIError(400, "WAV file is missing fmt/data chunks.", param, "invalid_value")
    audio_format, channels, rate, _, _, bits = fmt
    if audio_format == 0xFFFE:      # WAVE_FORMAT_EXTENSIBLE: trust the bit width
        audio_format = 3 if bits == 32 else 1
    if rate != AUDIO_SAMPLE_RATE:
        raise APIError(400, f"Audio must be {AUDIO_SAMPLE_RATE} Hz (got {rate}). "
                            "Resample at the capture edge.", param, "invalid_value")
    if audio_format == 1 and bits == 16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise APIError(400, f"Unsupported WAV encoding (format {audio_format}, {bits}-bit); "
                            "use PCM16 or float32.", param, "invalid_value")
    if channels > 1:
        samples = samples[:len(samples) - len(samples) % channels]
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def inkling_content_segments(content, param, audio_out):
    """Split OpenAI message content into ordered TMLv0 segments:
    ("text", str) for merged text runs, ("audio", n_frames) per input_audio
    part (its DMel bytes appended to audio_out in prompt order)."""
    if isinstance(content, str):
        return [("text", content)]
    if not isinstance(content, list):
        raise APIError(400, "Message content must be a string or an array of parts.", param)
    segments = []
    for index, part in enumerate(content):
        ptype = part.get("type") if isinstance(part, dict) else None
        if ptype in ("text", "input_text"):
            if not isinstance(part.get("text"), str):
                raise APIError(400, "Text content parts require a string `text` field.",
                               f"{param}.{index}.text")
            if segments and segments[-1][0] == "text":
                segments[-1] = ("text", segments[-1][1] + part["text"])
            else:
                segments.append(("text", part["text"]))
        elif ptype == "input_audio":
            spec = part.get("input_audio")
            if not isinstance(spec, dict) or not isinstance(spec.get("data"), str):
                raise APIError(400, "`input_audio` parts need base64 `data`.",
                               f"{param}.{index}.input_audio")
            if spec.get("format", "wav") != "wav":
                raise APIError(400, "Only WAV audio is supported (mono, 16 kHz, PCM16/float32).",
                               f"{param}.{index}.input_audio.format", "unsupported_content_type")
            import base64
            try:
                wav = base64.b64decode(spec["data"], validate=True)
            except Exception:
                raise APIError(400, "`input_audio.data` is not valid base64.",
                               f"{param}.{index}.input_audio.data")
            dmel = dmel_encode(decode_wav_mono16k(wav, f"{param}.{index}.input_audio.data"))
            audio_out.append(dmel)
            segments.append(("audio", len(dmel) // AUDIO_MEL_BANDS))
        else:
            raise APIError(400, "Unsupported content part type for the Inkling engine.",
                           f"{param}.{index}", "unsupported_content_type")
    return segments


def render_chat_kimi(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                     tool_choice=None):
    """Validated multi-turn K3 payload for the C engine.

    K3's rank-BPE makes ordinary-text segment boundaries part of the tokenizer
    contract. This private length-framed payload preserves roles, UTF-8 bytes,
    and message boundaries; kimi_k3.c constructs the native XTML tokens.
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or tool_choice not in (None, "none"):
        raise APIError(400, "Tool use is not wired up for the Kimi K3 engine yet.",
                       "tools", "unsupported_parameter")
    parts = ["K3CHAT1\n"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant"):
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        raw = message.get("content")
        text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
        reasoning = message.get("reasoning_content") if role == "assistant" else None
        if reasoning is not None and not isinstance(reasoning, str):
            raise APIError(400, "`reasoning_content` must be a string.",
                           f"messages.{index}.reasoning_content")
        if role == "assistant" and enable_thinking:
            reasoning = reasoning or ""
            parts.append(f"A {len(reasoning.encode('utf-8'))} {len(text.encode('utf-8'))}\n"
                         f"{reasoning or ''}{text}")
        else:
            parts.append(f"M {role} {len(text.encode('utf-8'))}\n{text}")
    parts.append(f"G {1 if enable_thinking else 0}\n")
    return "".join(parts)


def _build_v4_tools_section(tools, forced, tool_choice):
    """Build the DSML tools declaration section for the system prompt."""
    tools_section = [
        "## Tools",
        "",
        'You have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "'
        f"{DSML_TOOL_CALLS_OPEN}"
        f' block like the following:',
        "",
        DSML_TOOL_CALLS_OPEN,
        '<｜DSML｜invoke name="$TOOL_NAME">',
        '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>',
        "...",
        "</｜DSML｜invoke>",
        '<｜DSML｜invoke name="$TOOL_NAME2">',
        "...",
        "</｜DSML｜invoke>",
        DSML_TOOL_CALLS_CLOSE,
        "",
        'String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.',
        "",
        f"If thinking_mode is enabled (triggered by {THINK_OPEN}), you MUST output your complete reasoning inside {THINK_OPEN}...{THINK_CLOSE} BEFORE any tool calls or final response.",
        "",
        f"Otherwise, output directly after {THINK_CLOSE} with tool calls or final response.",
        "",
        "### Available Tool Schemas",
        "",
    ]
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
        tools_section.append(json.dumps(clean, ensure_ascii=False))
    tools_section.append("")
    tools_section.append("You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.")
    if forced:
        tools_section.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
    elif tool_choice == "required":
        tools_section.append("\n\nYou must call one of the functions above. Do not answer directly.")
    return "\n".join(tools_section)


def _render_v4_tool_calls(tool_calls):
    """Render a list of tool_calls into DSML format."""
    parts = []
    for tc in tool_calls:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        tc_name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = {}
        else:
            args = args_raw or {}
        parts.append("\n\n")
        parts.append(DSML_TOOL_CALLS_OPEN)
        parts.append("\n")
        parts.append(f'<{DSML}invoke name="{tc_name}">')
        parts.append("\n")
        for key, value in args.items():
            is_str = "true" if isinstance(value, str) else "false"
            val_str = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            parts.append(f'<{DSML}parameter name="{key}" string="{is_str}">{val_str}</{DSML}parameter>')
            parts.append("\n")
        parts.append(f"</{DSML}invoke>")
        parts.append("\n")
        parts.append(DSML_TOOL_CALLS_CLOSE)
    return "".join(parts)


def render_chat_v4(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                   tool_choice=None):
    """DeepSeek V4's native multi-turn chat template with tool calling support.

    Tool calls use DSML format: <｜DSML｜tool_calls> blocks with <｜DSML｜invoke>
    and <｜DSML｜parameter> tags. Tool results are wrapped in <tool_result>...</tool_result>
    and merged into user messages (the model was trained with tool results as part of
    the user turn, not as a separate role).
    """
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    user_tok = "<\uff5cUser\uff5c>"
    assistant_tok = "<\uff5cAssistant\uff5c>"
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"

    # Handle tool_choice: filter to forced tool or disable tools
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None

    parts = [bos]

    # Preprocess: merge tool messages into user messages
    # DeepSeek V4 doesn't have a "tool" role; tool results are <tool_result> blocks
    # inside user messages.
    if tools:
        merged = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise APIError(400, "Each message must be an object.", f"messages.{index}")
            role = message.get("role")
            if role not in ("system", "developer", "user", "assistant", "tool"):
                raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
            if role == "tool":
                tool_content = content_text(message.get("content"), f"messages.{index}.content")
                tool_block = f"{DSML_TOOL_RESULT_OPEN}{tool_content}{DSML_TOOL_RESULT_CLOSE}"
                # Merge into the last user message, or create a synthetic one
                if merged and merged[-1][0] == "user":
                    merged[-1][1] += tool_block
                else:
                    merged.append(("user", tool_block))
            else:
                merged.append((role, message))
    else:
        merged = [(m.get("role"), m) for m in messages]
        # Validate
        for index, (role, message) in enumerate(merged):
            if not isinstance(message, dict):
                raise APIError(400, "Each message must be an object.", f"messages.{index}")
            if role not in ("system", "developer", "user", "assistant"):
                raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")

    # Build tools section if needed
    tools_section = None
    if tools:
        tools_section = _build_v4_tools_section(tools, forced, tool_choice)

    for idx, (role, message) in enumerate(merged):
        if role in ("system", "developer"):
            text = content_text(message.get("content"), f"messages.{idx}.content") if message.get("content") is not None else ""
            parts.append(text)
            if tools_section:
                parts.append("\n\n" + tools_section)
        elif role == "user":
            if isinstance(message, str):
                # Merged tool result (just a string)
                parts.extend((user_tok, message))
            else:
                text = content_text(message.get("content"), f"messages.{idx}.content") if message.get("content") is not None else ""
                parts.extend((user_tok, text))
        elif role == "assistant":
            raw = message.get("content")
            text = content_text(raw, f"messages.{idx}.content") if raw is not None else ""
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise APIError(400, "`reasoning_content` must be a string.",
                               f"messages.{idx}.reasoning_content")
            parts.append(assistant_tok)
            if reasoning:
                parts.extend((THINK_OPEN, reasoning, THINK_CLOSE))
            else:
                parts.append(THINK_CLOSE)
            parts.append(text)
            # Render tool calls in DSML format (before EOS)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                parts.append(_render_v4_tool_calls(tool_calls))
            parts.append(eos)

    parts.extend((assistant_tok, THINK_OPEN if enable_thinking else THINK_CLOSE))
    return "".join(parts)


def render_chat_inkling(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                        tool_choice=None, audio_out=None):
    """Text-only subset of Inkling's chat_template.jinja: role tokens with
    <|content_text|> parts and <|end_message|> terminators, an assistant
    <|content_model_end_sampling|> after each prior model turn, the
    thinking-effort hint appended after the messages (the template's fallback
    branch), then <|message_model|> as the generation prompt."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    if tools or (tool_choice not in (None, "none")):
        raise APIError(400, "Tool use is not wired up for the Inkling engine yet.",
                       "tools", "unsupported_parameter")
    role_token = {"user": "<|message_user|>", "system": "<|message_system|>",
                  "developer": "<|message_system|>", "assistant": "<|message_model|>",
                  "tool": "<|message_tool|>"}
    # Thinking effort — template default is 0.9, but at single-machine decode
    # speeds unrequested reasoning burns the whole token budget before the answer
    # starts, so we default it OFF unless the client asks.
    effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
                  "high": 0.9, "max": 0.99}
    if reasoning_effort in effort_map:
        eff = effort_map[reasoning_effort]
    else:
        eff = 0.9 if enable_thinking else 0.0
    effort_str = ("<|message_system|><|content_text|>Thinking effort level: "
                  f"{0 if eff == 0.0 else eff}<|end_message|>")

    prompt = []
    effort_emitted = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        rtok = role_token.get(role)
        if rtok is None:
            raise APIError(400, f"Unsupported role {role!r}.", f"messages.{index}.role")
        # the template emits the effort hint inline, right before the first
        # non-system message — not at the end. Position matters: it changes the
        # exact token sequence the model was trained on.
        if not effort_emitted and role not in ("system", "developer"):
            prompt.append(effort_str)
            effort_emitted = True
        raw = message.get("content")
        if audio_out is not None and role == "user" and isinstance(raw, list):
            # multipart user content: text runs and audio clips become separate
            # TMLv0 messages, in part order (a message carries ONE content type).
            # Each DMel frame is one <|audio|> placeholder; the engine replaces
            # those embeddings with the frames appended to audio_out.
            for kind, val in inkling_content_segments(raw, f"messages.{index}.content", audio_out):
                if kind == "text":
                    prompt.append(f"{rtok}<|content_text|>{val}<|end_message|>")
                else:
                    prompt.append(f"{rtok}<|content_audio_input|>"
                                  + "<|audio|>" * val + "popup<|end_message|>")
        else:
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            prompt.append(f"{rtok}<|content_text|>{text}<|end_message|>")
        if role == "assistant":
            prompt.append("<|content_model_end_sampling|>")
    if not effort_emitted:                       # all-system edge case: fallback
        prompt.append(effort_str)
    prompt.append("<|message_model|>")           # add_generation_prompt
    # Thinking off: prefill the content channel. Without this the model can still
    # sample <|content_thinking|> as its first token (the effort hint is only a
    # soft signal), open a reasoning block, and burn the whole token budget before
    # reaching <|content_text|> — which the splitter then strips to an empty
    # answer. Ending the prompt at <|message_model|><|content_text|> forces content
    # mode; it is exactly the sequence every non-thinking turn is trained on.
    if eff == 0.0:
        prompt.append("<|content_text|>")
    return "".join(prompt)


def render_chat(messages, enable_thinking=False, reasoning_effort=None, tools=None,
                tool_choice=None):
    """Render the text-only subset of the official GLM-5.2 chat template."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    prompt = ["[gMASK]<sop>"]
    if enable_thinking:
        effort = "High" if reasoning_effort == "high" else "Max"
        prompt.append(f"<|system|>Reasoning Effort: {effort}")
    forced = None
    if isinstance(tool_choice, dict):
        forced = ((tool_choice.get("function") or {}).get("name")
                  or tool_choice.get("name"))
        if forced:
            tools = [t for t in (tools or [])
                     if ((t.get("function", t) if isinstance(t, dict) else {}).get("name") == forced)]
    elif tool_choice == "none":
        tools = None                              # the client forbade tools: do not offer them
    if tools:
        # AUTHORITATIVE GLM-5.2 tool-declaration block (byte-matches chat_template.jinja): the
        # `# Tools` + <tools></tools> XML structure is what the model was trained on. A made-up
        # preamble makes it hallucinate other frameworks' syntax (e.g. `end_action`).
        prompt.append("<|system|>\n# Tools\n\nYou may call one or more functions to assist with the "
                      "user query.\n\nYou are provided with function signatures within <tools></tools> "
                      "XML tags:\n<tools>\n")
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            clean = {k: v for k, v in fn.items() if k not in ("defer_loading", "strict")}
            prompt.append(json.dumps(clean, ensure_ascii=False) + "\n")
        prompt.append("</tools>\n\nFor each function call, output the function name and arguments "
                      "within the following XML format:\n<tool_call>{function-name}"
                      "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                      "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>")
        if forced:
            prompt.append(f"\n\nYou must call the function `{forced}`. Do not answer directly.")
        elif tool_choice == "required":
            prompt.append("\n\nYou must call one of the functions above. Do not answer directly.")
    prev_tool = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role in ("system", "developer"):
            prompt.append(f"<|system|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "user":
            prompt.append(f"<|user|>{content_text(message.get('content'), f'messages.{index}.content')}")
        elif role == "assistant":
            # content may be null when the message is purely tool_calls
            raw = message.get("content")
            text = content_text(raw, f"messages.{index}.content") if raw is not None else ""
            reasoning = message.get("reasoning_content")
            if reasoning is None:
                reasoning = ""
            elif not isinstance(reasoning, str):
                raise APIError(400, "`reasoning_content` must be a string.",
                               f"messages.{index}.reasoning_content")
            prompt.append(f"<|assistant|><think>{reasoning}</think>{text.strip()}")
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                prompt.append(BOX_START + (fn.get("name") or ""))
                for key, value in (args or {}).items():
                    prompt.append(f"<arg_key>{key}</arg_key><arg_value>"
                                  + (value if isinstance(value, str)
                                     else json.dumps(value, ensure_ascii=False)) + "</arg_value>")
                prompt.append(BOX_END)
        elif role == "tool":
            if not prev_tool:                       # one <|observation|> per consecutive tool run
                prompt.append("<|observation|>")
            prompt.append(TR_OPEN + content_text(message.get("content"), f"messages.{index}.content") + TR_CLOSE)
        else:
            raise APIError(400, f"Unsupported message role: {role!r}.",
                           f"messages.{index}.role", "unsupported_role")
        prev_tool = (role == "tool")
    prompt.append("<|assistant|><think>" if enable_thinking else
                  "<|assistant|><think></think>")
    return "".join(prompt)


# ---- Anthropic Messages API (#343) --------------------------------------------------------
# A translation layer, NOT a second engine path: /v1/messages rewrites an Anthropic-shaped
# request into the exact OpenAI-shaped body the existing path already validates, so prompt
# rendering, scheduling, generation and tool parsing stay single-sourced. Only the request
# translation and the response/SSE shapes are new. Claude Code is the reference client.

ANTHROPIC_LOCAL_SIGNATURE = "colibri-local"  # opaque compatibility metadata, not a crypto proof


class ThinkingStreamSplit:
    """Split GLM's reasoning marker without leaking markers across stream chunks."""
    MARKERS = (THINK_OPEN, THINK_CLOSE)

    def __init__(self, on_thinking, on_text, on_thinking_end=None, initial_thinking=True):
        self.on_thinking = on_thinking
        self.on_text = on_text
        self.on_thinking_end = on_thinking_end
        # #597: GLM emits reasoning only when the prompt opened <think> (thinking on);
        # with thinking off the prompt already closed it, so output is pure answer and
        # the splitter must start in text mode or it would file the whole answer as reasoning.
        self.thinking = initial_thinking
        self.buf = ""

    def _emit(self, text):
        if text:
            (self.on_thinking if self.thinking else self.on_text)(text)

    def feed(self, chunk):
        self.buf += chunk
        while True:
            hits = [(offset, marker) for marker in self.MARKERS
                    if (offset := self.buf.find(marker)) >= 0]
            if hits:
                offset, marker = min(hits, key=lambda hit: hit[0])
                self._emit(self.buf[:offset])
                self.buf = self.buf[offset + len(marker):]
                if marker == THINK_CLOSE and self.thinking:
                    self.thinking = False
                    if self.on_thinking_end:
                        self.on_thinking_end()
                continue

            hold = 0
            for size in range(1, min(len(self.buf), max(map(len, self.MARKERS)) - 1) + 1):
                if any(marker.startswith(self.buf[-size:]) for marker in self.MARKERS):
                    hold = size
            flush = len(self.buf) - hold
            if flush:
                self._emit(self.buf[:flush])
                self.buf = self.buf[flush:]
            return

    def finish(self):
        self._emit(self.buf)
        self.buf = ""

    close = finish        # interface parity with InklingStreamSplit in the streaming path


def split_thinking_reply(text, enable_thinking=True):
    """Return the marker-free (thinking, answer) portions of one GLM reply."""
    thinking, answer = [], []
    split = ThinkingStreamSplit(thinking.append, answer.append, initial_thinking=enable_thinking)
    split.feed(text)
    split.finish()
    return "".join(thinking), "".join(answer)


def _anthropic_block_text(blocks, param):
    """Text out of an Anthropic content array (tool_result content is the same shape)."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        raise APIError(400, "Content must be a string or an array of blocks.", param)
    parts = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise APIError(400, "Colibri currently supports text blocks only here.",
                           f"{param}.{index}", "unsupported_content_type")
        if not isinstance(block.get("text"), str):
            raise APIError(400, "Text blocks require a string `text` field.", f"{param}.{index}.text")
        parts.append(block["text"])
    return "".join(parts)


def anthropic_to_openai(body):
    """Anthropic request -> (messages, tools, tool_choice) in OpenAI shape."""
    messages = []
    system = body.get("system")
    if isinstance(system, str):
        if system:
            messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _anthropic_block_text(system, "system")
        if text:
            messages.append({"role": "system", "content": text})
    elif system is not None:
        raise APIError(400, "`system` must be a string or an array of text blocks.", "system")

    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise APIError(400, "`messages` must be a non-empty array.", "messages")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise APIError(400, "Each message must be an object.", f"messages.{index}")
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise APIError(400, f"Input message role {role!r} is not supported. Anthropic messages are "
                           "`user` or `assistant`; a system prompt goes in the top-level `system`.",
                           f"messages.{index}.role", "unsupported_role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise APIError(400, "Message content must be a string or an array of blocks.",
                           f"messages.{index}.content")
        texts, reasoning, calls, results = [], [], [], []
        for j, block in enumerate(content):
            where = f"messages.{index}.content.{j}"
            if not isinstance(block, dict):
                raise APIError(400, "Each content block must be an object.", where)
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise APIError(400, "Text blocks require a string `text` field.", f"{where}.text")
                texts.append(block["text"])
            elif kind == "thinking":
                if role != "assistant":
                    raise APIError(400, "`thinking` blocks are valid only in assistant messages.",
                                   f"{where}.type", "unsupported_content_type")
                if not isinstance(block.get("thinking"), str):
                    raise APIError(400, "Thinking blocks require a string `thinking` field.",
                                   f"{where}.thinking")
                if not isinstance(block.get("signature"), str):
                    raise APIError(400, "Thinking blocks require a string `signature` field.",
                                   f"{where}.signature")
                reasoning.append(block["thinking"])
            elif kind == "tool_use":
                name = block.get("name")
                if not isinstance(name, str) or not name:
                    raise APIError(400, "`tool_use` blocks require a string `name`.", f"{where}.name")
                arguments = block.get("input")
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    raise APIError(400, "`tool_use.input` must be an object.", f"{where}.input")
                calls.append({"id": block.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
                              "type": "function",
                              "function": {"name": name,
                                           "arguments": json.dumps(arguments, ensure_ascii=False)}})
            elif kind == "tool_result":
                results.append({"role": "tool",
                                "tool_call_id": block.get("tool_use_id") or "",
                                "content": _anthropic_block_text(block.get("content", ""),
                                                                 f"{where}.content")})
            else:
                raise APIError(400, "Colibri supports `text`, `tool_use` and `tool_result` "
                               "content blocks only.", f"{where}.type", "unsupported_content_type")
        # tool results precede the user's own text: they answer the previous assistant turn
        messages.extend(results)
        text = "".join(texts)
        if role == "assistant":
            if text or reasoning or calls:
                entry = {"role": "assistant", "content": text or None}
                if reasoning:
                    entry["reasoning_content"] = "".join(reasoning)
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
        elif text or not results:
            messages.append({"role": "user", "content": text})
    return messages


def anthropic_tools(body):
    """Anthropic tools/tool_choice -> OpenAI shape (validated downstream by generation_options)."""
    raw = body.get("tools")
    if raw is None:
        tools = None
    elif not isinstance(raw, list):
        raise APIError(400, "`tools` must be an array.", "tools")
    else:
        tools = []
        for index, tool in enumerate(raw):
            if not isinstance(tool, dict):
                raise APIError(400, "Each tool must be an object.", f"tools.{index}")
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise APIError(400, "Each tool requires a string `name`.", f"tools.{index}.name")
            schema = tool.get("input_schema")
            if schema is not None and not isinstance(schema, dict):
                raise APIError(400, "`input_schema` must be an object.", f"tools.{index}.input_schema")
            function = {"name": name, "parameters": schema or {"type": "object", "properties": {}}}
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            tools.append({"type": "function", "function": function})
        return tools, (body.get("tool_choice") or {}).get("name")
    return tools, None


def parse_stop_sequences(body):
    value = body.get("stop")
    if value is None:
        return ()
    if isinstance(value, str):
        sequences = [value]
    elif isinstance(value, list):
        sequences = value
    else:
        raise APIError(400, "`stop` must be a string or an array of strings.",
                       "stop", "invalid_value")
    if not 1 <= len(sequences) <= 4:
        raise APIError(400, "`stop` must contain between 1 and 4 sequences.",
                       "stop", "invalid_value")
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str) or not sequence:
            raise APIError(400, "Each `stop` sequence must be a non-empty string.",
                           f"stop.{index}", "invalid_value")
    return tuple(sequences)


def conversation_cache_slot(messages, kv_slots):
    """Stable KV slot for a conversation so its turns reuse the same cached prefix.

    The chat APIs are stateless: every turn resends the whole history, and the engine
    caches each KV slot's prefix. When the client does not pin a `cache_slot`, the
    scheduler falls back to `min(free_slots)`, which is blind to which slot already
    holds this conversation. Under any interleaving of clients a turn can then land on
    another conversation's slot and force a full re-prefill (#634, Defect 1). Hashing a
    key that stays constant across a conversation's turns — the leading system messages
    plus the first user message, which never change once the conversation has started —
    routes every turn of one conversation to the same slot. Distinct conversations
    spread across slots; when there are more live conversations than slots, colliding
    ones degrade to the old re-prefill behaviour rather than to anything worse.

    Returns a slot in [0, kv_slots). Falls back to 0 when there is nothing to key on.
    """
    if kv_slots <= 1 or not isinstance(messages, list) or not messages:
        return 0
    prefix = []
    for message in messages:
        prefix.append(message)
        if isinstance(message, dict) and message.get("role") == "user":
            break                 # first user turn reached: the key is now stable for the whole conversation
    try:
        key = json.dumps(prefix, sort_keys=True, default=str)
    except (TypeError, ValueError):
        key = repr(prefix)
    digest = hashlib.sha1(key.encode("utf-8", "replace")).digest()
    return int.from_bytes(digest[:8], "big") % kv_slots


DEFAULT_CHAT_STOP_SEQUENCES = (
    "<|assistant|>", "<|user|>", "<|system|>", "<|observation|>",
    "<｜User｜>", "<｜Assistant｜>",
)


def stop_policy(body, chat):
    sequences = parse_stop_sequences(body)
    ignore_leading = body.get("x_colibri_ignore_leading_stop", False)
    if not isinstance(ignore_leading, bool):
        raise APIError(400, "`x_colibri_ignore_leading_stop` must be a boolean.",
                       "x_colibri_ignore_leading_stop", "invalid_value")
    if chat and ARCH == "glm" and not sequences:
        # The GLM chat template owns these role boundaries, so generic OpenAI
        # clients should not need model-specific stop knowledge. Inkling has a
        # different marker family and receives no implicit GLM stops. Treat an
        # occasional leading GLM marker patiently; client-provided stops remain
        # strict unless the extension is explicitly requested.
        return DEFAULT_CHAT_STOP_SEQUENCES, True
    return sequences, ignore_leading


class StopFilter:
    """Stream text without exposing a full or partial stop sequence."""
    def __init__(self, sequences, emit, ignore_leading=False):
        self.sequences = tuple(sequences)
        self.emit = emit
        self.ignore_leading = ignore_leading
        self.pending = ""
        self.matched = None
        self.useful_content_seen = False
        self.leading_matches_ignored = 0

    def _emit(self, text):
        if text:
            self.emit(text)
            if text.strip():
                self.useful_content_seen = True

    def feed(self, chunk):
        if self.matched is not None:
            return
        text = self.pending + chunk
        self.pending = ""
        while True:
            match = None
            for order, sequence in enumerate(self.sequences):
                offset = text.find(sequence)
                candidate = (offset, order, sequence)
                if offset >= 0 and (match is None or candidate[:2] < match[:2]):
                    match = candidate
            if match is None:
                break
            offset, _order, sequence = match
            prefix = text[:offset]
            if (self.ignore_leading and not self.useful_content_seen
                    and not prefix.strip()):
                self.leading_matches_ignored += 1
                text = text[offset + len(sequence):]
                if not text:
                    return
                continue
            self.matched = sequence
            self._emit(prefix)
            return

        hold = 0
        maximum = min(len(text), max((len(s) - 1 for s in self.sequences), default=0))
        for size in range(1, maximum + 1):
            suffix = text[-size:]
            if any(sequence.startswith(suffix) for sequence in self.sequences):
                hold = size
        flush = len(text) - hold
        if flush:
            self._emit(text[:flush])
        self.pending = text[flush:]

    def finish(self):
        if self.matched is None and self.pending:
            self._emit(self.pending)
        self.pending = ""

    def stopped(self):
        return self.matched is not None


GENERIC_JSON_GBNF = r"""root ::= element
element ::= object | array | string | number | boolean | null
object ::= "{" ws (string ":" ws element ("," ws string ":" ws element)*)? "}"
array ::= "[" ws (element ("," ws element)*)? "]"
string ::= "\"" (esc | [^"\\\x00-\x1f])* "\""
esc ::= "\\"" | "\\/" | "\\\\" | "\\n" | "\\t" | "\\r" | "\\b" | "\\f" | "\\u" [0-9a-fA-F]{4}
number ::= "-"? ([0-9]+ | "0") ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \t\n\r]*
"""


def generation_options(body, limit):
    if body.get("n", 1) != 1:
        raise APIError(400, "Colibri currently supports `n=1` only.", "n", "unsupported_value")
    # `tools`/`functions` are handled by render_chat (declaration) + parse_tool_calls (output).
    # Validate tools/functions structure early so malformed input fails with a clear error.
    tools_raw = body.get("tools") or body.get("functions")
    if tools_raw is not None:
        if not isinstance(tools_raw, list):
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        if not tools_raw:
            raise APIError(400, "`tools` must be a non-empty array.", "tools", "invalid_value")
        for idx, tool in enumerate(tools_raw):
            if not isinstance(tool, dict):
                raise APIError(400, f"Each tool must be an object, got {type(tool).__name__} at index {idx}.",
                               f"tools.{idx}", "invalid_value")
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            if not isinstance(fn, dict):
                raise APIError(400, f"Tool function must be an object at index {idx}.",
                               f"tools.{idx}.function", "invalid_value")
            if not fn.get("name"):
                raise APIError(400, f"Each tool must have a `name` at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
            if not isinstance(fn["name"], str):
                raise APIError(400, f"Tool `name` must be a string at index {idx}.",
                               f"tools.{idx}.function.name", "invalid_value")
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            if choice not in ("auto", "none", "required"):
                raise APIError(400, "`tool_choice` must be one of \"auto\", \"none\", \"required\", "
                                    "or a function object.", "tool_choice", "unsupported_value")
        elif isinstance(choice, dict):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise APIError(400, "`tool_choice` function object must include a name.",
                               "tool_choice", "invalid_value")
            declared = [(t.get("function", t) if isinstance(t, dict) else {}).get("name")
                        for t in (body.get("tools") or body.get("functions") or [])]
            if name not in declared:
                raise APIError(400, f"`tool_choice` names {name!r}, which is not in `tools`.",
                               "tool_choice", "invalid_value")
        else:
            raise APIError(400, "`tool_choice` must be a string or a function object.",
                           "tool_choice", "invalid_value")
        if choice != "none" and not (body.get("tools") or body.get("functions")):
            raise APIError(400, "`tool_choice` requires `tools`.", "tool_choice", "invalid_value")
    stop_sequences = parse_stop_sequences(body)
    if body.get("logprobs"):
        raise APIError(400, "Log probabilities are not supported yet.", "logprobs", "unsupported_parameter")
    if body.get("frequency_penalty", 0) or body.get("presence_penalty", 0):
        raise APIError(400, "Token penalties are not supported yet.", None, "unsupported_parameter")
    if body.get("seed") is not None:
        raise APIError(400, "Per-request seeds are not supported yet.", "seed", "unsupported_parameter")
    # response_format -> optional per-request grammar for the engine's grammar-forced
    # draft source (#70/#148). NEVER a sampling constraint: drafts are verified, so a
    # schema the engine cannot compile degrades to "no speedup", not to an error and
    # not to changed output. json_schema payloads are forwarded as-is (the engine
    # compiles them via schema_gbnf.h); {"type": "gbnf"} is a raw-GBNF extension.
    grammar = None
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        if not isinstance(response_format, dict) or "type" not in response_format:
            raise APIError(400, "`response_format` must be an object with a `type`.",
                           "response_format", "invalid_value")
        ftype = response_format["type"]
        if ftype == "json_object":
            grammar = GENERIC_JSON_GBNF
        elif ftype == "json_schema":
            schema = response_format.get("json_schema")
            if isinstance(schema, dict):
                grammar = json.dumps(schema, ensure_ascii=False)
        elif ftype == "gbnf":
            grammar = response_format.get("value")
        else:
            raise APIError(400, f"Unknown response_format type {ftype!r}.",
                           "response_format.type", "unsupported_value")
    if grammar is not None and not isinstance(grammar, str):
        raise APIError(400, "`response_format` grammar must be a string.",
                       "response_format", "invalid_value")
    maximum = body.get("max_tokens")
    if maximum is not None and not isinstance(maximum, int):
        raise APIError(400, "`max_tokens` must be an integer.", "max_tokens", "invalid_value")
    if maximum is not None and maximum < 1:
        raise APIError(400, "`max_tokens` must be at least 1.", "max_tokens", "invalid_value")
    if maximum is None:
        maximum = limit
    temperature = body.get("temperature")
    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise APIError(400, "`temperature` must be a number.", "temperature", "invalid_value")
        if temperature < 0 or temperature > 2:
            raise APIError(400, "`temperature` must be between 0 and 2.", "temperature", "invalid_value")
    else:
        temperature = 1.0
    top_p = body.get("top_p")
    if top_p is not None:
        if not isinstance(top_p, (int, float)):
            raise APIError(400, "`top_p` must be a number.", "top_p", "invalid_value")
        if top_p < 0 or top_p > 1:
            raise APIError(400, "`top_p` must be between 0 and 1.", "top_p", "invalid_value")
    else:
        top_p = 1.0
    return maximum, temperature, top_p, grammar, stop_sequences


def model_arch(model_id):
    """Detect the engine architecture from the model file name or path."""
    model_type = model_id.lower()
    if "inkling" in model_type:
        return "inkling"
    if "kimi" in model_type or "k3" in model_type:
        return "kimi"
    if "deepseek_v4" in model_type or ("deepseek" in model_type and "v4" in model_type):
        return "deepseek_v4"
    if "glm" in model_type or "chatglm" in model_type:
        return "glm"
    return "glm"


def _tune_child_env(env, arch):
    """Apply engine-local defaults for OMP and speculative decoding."""
    if arch != "deepseek_v4":
        return env
    if env.get("COLI_NO_OMP_TUNE"):
        return env
    try:
        from resource_plan import physical_cpu_count
        cpu_count = physical_cpu_count()
    except Exception:
        cpu_count = os.cpu_count() or 4
    env.setdefault("OMP_NUM_THREADS", str(cpu_count))
    env.setdefault("OMP_WAIT_POLICY", "active")
    env.setdefault("GOMP_SPINCOUNT", "200000")
    env.setdefault("OMP_DYNAMIC", "FALSE")
    if sys.platform != "win32":
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
    env.setdefault("V4_DRAFT", "0")
    env.setdefault("V4_MTP", "0")
    env.setdefault("V4_MTP_DRAFT", "3")
    env.setdefault("V4_MTP_GB", "0.45")
    env.setdefault("V4_MTP_MISS", "96")
    env.setdefault("V4_MTP_MIN", "3")
    env.setdefault("V4_MTP_CONF", "0.55")
    return env


def _cap_for_arch(arch, cap):
    """Resolve the cap sentinel for the GLM engine.

    cap=None means "auto": 0 for GLM (platform-aware), 8 for other arches.
    An explicit int (including 0) passes through verbatim."""
    if cap is not None:
        return cap
    return 0 if arch == "glm" else 8


class Engine:
    """Thin wrapper around the subprocess lifecycle."""

    def __init__(self, path, model, ctx, max_tokens, max_batch, kv_slots,
                 gpu_layer=0, tensor_split=None, mlock=False, flash_attn=False,
                 threads=0, threads_batch=0, verbose=False, cap=None,
                 model_arch="glm", env=None):
        self.arch = model_arch
        self.path = path
        self.model = model
        self.ctx = ctx
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.kv_slots = kv_slots
        self.gpu_layer = gpu_layer
        self.tensor_split = tensor_split
        self.mlock = mlock
        self.flash_attn = flash_attn
        self.threads = threads
        self.threads_batch = threads_batch
        self.verbose = verbose
        self.cap = cap
        self._base_env = dict(env or os.environ)
        self._process = None
        self._rpipe = None
        self._wpipe = None
        self._lock = threading.Lock()
        self._closed = False
        self._stats = {}

    def start(self):
        if self._process:
            return
        args = [str(self.path)]
        # GLM engine takes a positional cap argument
        if self.arch == "glm":
            args.append(str(_cap_for_arch(self.arch, self.cap)))

        # Engines read configuration from environment variables, not CLI args
        child_env = dict(self._base_env,
                         SNAP=str(self.model),
                         SERVE="1",
                         SERVE_BATCH="1",
                         NGEN=str(self.max_tokens),
                         KV_SLOTS=str(self.kv_slots))
        # deepseek_v4 also reads CTX from env
        if self.arch in ("inkling", "kimi", "deepseek_v4"):
            child_env["CTX"] = str(self.ctx)
        # Apply OMP tuning for deepseek_v4
        _tune_child_env(child_env, self.arch)

        self._process = subprocess.Popen(
            args, env=child_env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, cwd=str(HERE))
        self._rpipe = self._process.stdout
        self._wpipe = self._process.stdin
        self._next_id = 1
        # Tee engine stderr to our stderr so crash messages are visible
        self._stderr_thread = threading.Thread(
            target=self._tee_stderr, daemon=True)
        self._stderr_thread.start()

    def _tee_stderr(self):
        """Forward engine stderr to our stderr so crash/OOM messages are visible."""
        try:
            for line in self._process.stderr:
                sys.stderr.write(f"[engine] {line.decode('utf-8', errors='replace')}")
                sys.stderr.flush()
        except OSError:
            pass

    def _check_alive(self):
        """Check if the engine subprocess is still running."""
        rc = self._process.poll()
        if rc is not None:
            raise RuntimeError(f"Engine process exited with code {rc}")

    @property
    def _request_id(self):
        rid = self._next_id
        self._next_id += 1
        return rid

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._wpipe:
            try:
                self._wpipe.close()
            except OSError:
                pass
        if self._rpipe:
            try:
                self._rpipe.close()
            except OSError:
                pass
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=3)
                except Exception:
                    pass

    def generate(self, prompt, max_tokens, temperature, top_p, on_chunk,
                 cache_slot, cancelled, grammar=None, stopped=None,
                 on_accept=None, audio=None):
        """Send a generation request to the engine and return stats.

        The engine writes DATA frames (decoded text) to stdout; the caller
        receives each chunk through `on_chunk`.  `cancelled` is a callable
        returning True when the client disconnected (to abort long generations).
        `on_accept` is called with the ACCEPT frame payload once the prompt
        is accepted (before prefill starts).
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed")
            self._check_alive()
            if self.arch == "glm":
                # GLM protocol: SUBMIT header with 6 fields, then prompt body
                header = (f"SUBMIT {self.ctx} {max_tokens} {temperature} "
                          f"{top_p} {cache_slot}\n")
                if grammar is not None:
                    header += f"GRAMMAR {len(grammar.encode('utf-8'))}\n"
                self._wpipe.write(header.encode("utf-8"))
                self._wpipe.write(prompt.encode("utf-8"))
                self._wpipe.write(END)
            elif self.arch == "deepseek_v4":
                # deepseek_v4 protocol: SUBMIT <id> <slot> <prompt_bytes> <max_tokens>
                # <temperature> <top_p> [extension_bytes]\n<payload>\n
                prompt_bytes = prompt.encode("utf-8")
                rid = self._request_id
                ext_len = len(audio) if audio else 0
                v4_slot = cache_slot if cache_slot is not None else 0
                header = (f"SUBMIT {rid} {v4_slot} {len(prompt_bytes)} "
                          f"{max_tokens} {temperature} {top_p} {ext_len}\n")
                sys.stderr.write(f"[engine] SUBMIT id={rid} slot={v4_slot} "
                                 f"bytes={len(prompt_bytes)} max_tok={max_tokens} "
                                 f"temp={temperature} top_p={top_p} ext={ext_len}\n")
                sys.stderr.flush()
                self._wpipe.write(header.encode("utf-8"))
                self._wpipe.write(prompt_bytes)
                if audio is not None:
                    self._wpipe.write(audio)
                self._wpipe.write(b"\n")  # delimiter (C code does fgetc(stdin))
                sys.stderr.write(f"[engine] SUBMIT write complete, waiting for response...\n")
                sys.stderr.flush()
            elif self.arch in ("inkling", "kimi"):
                # Colibri protocol: SUBMIT with 6 fields, then prompt body
                header = (f"SUBMIT {self.ctx} {max_tokens} {temperature} "
                          f"{top_p} {cache_slot}\n")
                if grammar is not None:
                    header += f"GRAMMAR {len(grammar.encode('utf-8'))}\n"
                if audio is not None:
                    header += f"AUDIO {len(audio)}\n"
                    self._wpipe.write(audio)
                self._wpipe.write(header.encode("utf-8"))
                self._wpipe.write(prompt.encode("utf-8"))
                self._wpipe.write(END)
            else:
                raise ValueError(f"Unknown architecture: {self.arch}")

        # Read response frames
        buf = b""
        stats = {"prompt_tokens": 0, "completion_tokens": 0, "length_limited": False}
        read_count = 0
        while True:
            try:
                chunk = self._rpipe.read(65536)
            except OSError as e:
                sys.stderr.write(f"[engine] read error: {e}\n")
                sys.stderr.flush()
                break
            if not chunk:
                rc = self._process.poll()
                sys.stderr.write(f"[engine] pipe closed (process exited, rc={rc})\n")
                sys.stderr.flush()
                break
            read_count += 1
            sys.stderr.write(f"[engine] read #{read_count}: got {len(chunk)} bytes, total buf={len(buf)+len(chunk)}\n")
            sys.stderr.flush()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.rstrip(b"\r")
                sys.stderr.write(f"[engine] line: {line[:120]}\n")
                sys.stderr.flush()
                if self.arch == "deepseek_v4":
                    # deepseek_v4 uses DONE frame (not END sentinel)
                    # Format: DONE <id> STAT <completion> <tps> <hit_rate> <rss>
                    #         <prompt_tokens> <length_limited> <prefix_reused>
                    # Indices:  0     1   2      3            4      5         6
                    #           7               8                  9
                    if line.startswith(b"DONE "):
                        parts = line.decode("utf-8", errors="replace").split()
                        # Extract stats from DONE frame if available
                        if len(parts) >= 9:
                            try:
                                stats["completion_tokens"] = int(parts[3])
                                stats["prompt_tokens"] = int(parts[7])
                                stats["length_limited"] = parts[8] == "1"
                            except (ValueError, IndexError):
                                pass
                        sys.stderr.write(f"[engine] DONE: {line.decode('utf-8', errors='replace')}\n")
                        sys.stderr.flush()
                        return stats
                    if line == READY:
                        continue
                    if line.startswith(b"ACCEPT "):
                        sys.stderr.write(f"[engine] ACCEPT received: {line.decode('utf-8', errors='replace')}\n")
                        sys.stderr.flush()
                        on_accept and on_accept(line[7:])
                        continue
                    if line.startswith(b"DATA "):
                        # C engine writes: "DATA <id> <bytes>\n<data bytes>\n"
                        # After split on \n, `line` is the header, `buf` starts with data.
                        parts = line.decode("utf-8", errors="replace").split()
                        if len(parts) >= 3:
                            data_len = int(parts[2])
                            # Ensure we have enough bytes in the buffer
                            while len(buf) < data_len:
                                try:
                                    more = self._rpipe.read(65536)
                                    if not more:
                                        break
                                    buf += more
                                except OSError:
                                    break
                            if len(buf) >= data_len:
                                data = buf[:data_len]
                                buf = buf[data_len:]
                                # Skip trailing \n if present
                                if buf.startswith(b"\n"):
                                    buf = buf[1:]
                                text = data.decode("utf-8", errors="replace")
                                on_chunk(text)
                                stats["completion_tokens"] += 1
                    elif line.startswith(b"PROF "):
                        parts = line[5:].decode().split()
                        if len(parts) >= 3:
                            stats["prompt_tokens"] = int(parts[0])
                            stats["completion_tokens"] = int(parts[1])
                            stats["length_limited"] = parts[2] == "length"
                    elif line.startswith(b"STAT "):
                        # Startup STAT line from engine, silently ignored
                        continue
                    elif line.startswith(b"ERROR "):
                        msg = line[6:].decode("utf-8", errors="replace")
                        sys.stderr.write(f"[engine] ERROR: {msg}\n")
                        sys.stderr.flush()
                        fields = msg.split()
                        raise _engine_error(fields, msg)
                    elif cancelled and cancelled():
                        self._wpipe.write(b"CANCEL\n")
                        while True:
                            try:
                                leftover = self._rpipe.read(65536)
                            except OSError:
                                break
                            if not leftover:
                                break
                            if b"DONE " in leftover or b"ERROR " in leftover:
                                break
                        return stats
                else:
                    # GLM / inkling / kimi protocol (END-based)
                    if line == END:
                        return stats
                    if line == READY:
                        continue
                    if line.startswith(b"ACCEPT "):
                        on_accept and on_accept(line[7:])
                        continue
                    if line.startswith(b"DATA "):
                        payload = line[5:]
                        on_chunk(payload.decode("utf-8", errors="replace"))
                        stats["completion_tokens"] += 1
                    elif line.startswith(b"PROF "):
                        parts = line[5:].decode().split()
                        if len(parts) >= 3:
                            stats["prompt_tokens"] = int(parts[0])
                            stats["completion_tokens"] = int(parts[1])
                            stats["length_limited"] = parts[2] == "length"
                    elif line.startswith(b"ERROR "):
                        msg = line[6:].decode("utf-8", errors="replace")
                        fields = msg.split()
                        raise _engine_error(fields, msg)
                    elif cancelled and cancelled():
                        self._wpipe.write(b"CANCEL\n")
                        # drain the rest of the response
                        while True:
                            try:
                                leftover = self._rpipe.read(65536)
                            except OSError:
                                break
                            if not leftover:
                                break
                            if END in leftover:
                                break
                        return stats
        sys.stderr.write(f"[engine] generate exiting with stats: {stats}\n")
        sys.stderr.flush()
        return stats


class ColibriHandler(BaseHTTPRequestHandler):
    """HTTP handler for the OpenAI-compatible API."""

    def log_message(self, format, *args):
        try:
            level = int(os.environ.get("COLI_LOG", "1"))
        except ValueError:
            level = 1
        if level >= 1:
            sys.stderr.write("%s - - [%s] %s\n" %
                             (self.client_address[0], self.log_date_time_string(),
                              format % args))
            sys.stderr.flush()

    def send_cors_headers(self):
        origin = self.headers.get("Origin", "")
        if origin in DEFAULT_CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods",
                             "GET, POST, OPTIONS, DELETE")
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type, X-Request-ID")
            self.send_header("Access-Control-Max-Age", "86400")

    def send_json(self, status, body, request_id=None, extra_headers=None):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if request_id:
            self.send_header("x-request-id", request_id)
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            raise APIError(413, f"Request body too large (max {MAX_BODY} bytes).",
                           None, "request_too_large")
        return self.rfile.read(length)

    def _parse_json_body(self):
        try:
            return json.loads(self._read_body())
        except json.JSONDecodeError as exc:
            raise APIError(400, f"Invalid JSON: {exc}", None, "invalid_json")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/v1/models" or path == "/models":
            self.send_json(200, {"object": "list", "data": [{"id": self.server.model_id,
                            "object": "model", "owned_by": "colibri"}]})
        elif path == "/health":
            self.send_json(200, {"status": "ok"})
        elif path == "/profile":
            self.send_json(200, self.server._profile())
        elif path == "/queue":
            self.send_json(200, self.server.scheduler.snapshot())
        else:
            self._fail(APIError(404, "Not found.", None, "not_found"), self.headers.get("x-request-id"))

    def do_POST(self):
        path = urlsplit(self.path).path
        request_id = self.headers.get("x-request-id", "") or str(uuid.uuid4())
        try:
            body = self._parse_json_body()
        except APIError as exc:
            self._fail(exc, request_id)
            return

        if path == "/v1/chat/completions" or path == "/chat/completions":
            self.chat_completion(body, request_id)
        elif path == "/v1/completions" or path == "/completions":
            self.completion(body, request_id)
        elif path == "/v1/messages":
            self.anthropic_messages(body, request_id)
        else:
            self._fail(APIError(404, "Not found.", None, "not_found"), request_id)

    def _fail(self, error, request_id):
        """Report an error, unless the response is already on the wire. Once a streaming 200
        is committed, a second status line would be framed as SSE body -- clients saw a whole
        `HTTP/1.1 500` spliced into the event stream. All we can still do is stop talking; the
        stream ends at the close, which the 200 already announced (#597 item 3)."""
        if self._committed:
            self.close_connection = True
            return
        self.send_json(error.status, self.error_body(error), request_id, error.headers)

    def error_body(self, error):
        """Anthropic clients parse a different error envelope; the OpenAI one is unchanged."""
        if urlsplit(self.path).path != "/v1/messages":
            return error_object(error)
        return {"type": "error", "error": {"type": error.error_type, "message": error.message}}

    def generation(self, body, prompt, request_id, chat, tools=None, tool_choice=None,
                   enable_thinking=False, audio=None):
        # COLI_DEBUG tees the engine transaction to stderr: 1 = decoded output stream only,
        # 2 = both sides (rendered prompt + output). render_chat already folds prior turns and
        # tool results into `prompt`, so level 2 is the full conversation the engine saw.
        try:
            dbg = int(os.environ.get("COLI_DEBUG", "0"))
        except ValueError:
            dbg = 0
        if dbg >= 2:
            sys.stderr.write(f"\n===== PROMPT [{request_id}] =====\n{prompt}\n===== OUTPUT [{request_id}] =====\n")
            sys.stderr.flush()
        maximum, temperature, top_p, grammar, _requested_stop_sequences = generation_options(
            body, self.server.max_tokens)
        if grammar is not None and ARCH in ("inkling", "kimi"):
            # sibling engines speak the 6-field SUBMIT header only; sending the
            # grammar payload extension would desync its stdin framing.
            raise APIError(400, f"`response_format` grammars are not supported by the {ARCH} "
                                "engine yet.", "response_format", "unsupported_parameter")
        stop_sequences, ignore_leading_stop = stop_policy(body, chat)
        # tools and tool_choice come from chat_completion() already processed/filtered
        if chat and tool_choice == "none":
            tools = None          # client forbade tools: never surface tool_calls
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        if cache_slot is None and self.server.kv_slots > 1:
            # #634: pin each conversation to a stable KV slot so multi-turn reuses its
            # cached prefix instead of re-prefilling. Only when the request carries a
            # conversation; raw /v1/completions keeps the scheduler's free-slot pick.
            conversation = body.get("messages")
            if isinstance(conversation, list) and conversation:
                cache_slot = conversation_cache_slot(conversation, self.server.kv_slots)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        stream_options = body.get("stream_options") if stream else None
        if stream and stream_options is not None and not isinstance(stream_options, dict):
            raise APIError(400, "`stream_options` must be an object.", "stream_options")
        include_usage = bool((stream_options or {}).get("include_usage"))
        object_name = "chat.completion" if chat else "text_completion"
        id_prefix = "chatcmpl-" if chat else "cmpl-"
        completion_id = id_prefix + uuid.uuid4().hex
        created = int(time.time())

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stop_filter = StopFilter(stop_sequences, output.append, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped,
                    **({"audio": audio} if audio else {}))
                stop_filter.finish()
                text = "".join(output)
                reasoning = ""
                if ARCH == "inkling":
                    text, reasoning = split_inkling(text)
                elif chat:
                    # #597 item 4: GLM emits reasoning then </think> then the answer. Route the
                    # reasoning to reasoning_content instead of dumping it (or the raw </think>)
                    # into the visible answer / tool-call parser.
                    reasoning, text = split_thinking_reply(text, enable_thinking)
                length_finish = "length" if stats["length_limited"] else "stop"
                if chat and tools:
                    # Use v4 parser for DeepSeek V4, GLM parser for others
                    if ARCH == "deepseek_v4":
                        content, calls = parse_tool_calls_v4(text, tools)
                    else:
                        content, calls = parse_tool_calls(text, tools)
                    message = {"role": "assistant", "content": content or None, "refusal": None}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    if calls:
                        message["tool_calls"] = calls
                    finish = "tool_calls" if calls else length_finish
                    choice = {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                else:
                    _msg = {"role": "assistant", "content": text, "refusal": None}
                    if reasoning:
                        _msg["reasoning_content"] = reasoning
                    choice = ({"index": 0, "message": _msg,
                               "logprobs": None, "finish_reason": length_finish} if chat else
                              {"index": 0, "text": text, "logprobs": None, "finish_reason": length_finish})
                self.send_json(200, {"id": completion_id, "object": object_name, "created": created,
                    "model": self.server.model_id, "choices": [choice], "usage": self.usage(stats)},
                    request_id, queue_headers)
                return

            stream_object = "chat.completion.chunk" if chat else object_name
            # #597 item 6: DO NOT commit the 200 yet. The engine validates the prompt against the
            # context AFTER we would have sent headers, so an oversized prompt used to be a
            # CONTEXT_EXCEEDED discovered too late to send a clean 400. Defer the SSE headers into
            # start_stream(), fired on the engine's ACCEPT frame (before prefill); an ERROR that
            # arrives before ACCEPT propagates as an APIError with nothing committed -> proper 400.
            connected = False
            stream_started = [False]
            ka_thread = [None]
            # KEEPALIVE: engine.generate() blocks SILENTLY during the (minutes-long) cold
            # prefill, and the client drops the socket after its idle timeout. A background pump
            # emits a keepalive delta whenever no event has been written for KA_GAP seconds. All
            # wfile writes share ka_lock so the pump and event() never interleave; last_write
            # gates the pump so it stays quiet while real tokens are flowing (e.g. during decode).
            ka_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()
            KA_GAP = 10.0
            dbg_echo = dbg >= 1   # tee decoded tokens to stderr (COLI_DEBUG level parsed in generation())

            def event(choices, usage_marker=False):
                nonlocal connected
                if not connected:
                    return
                event_body = {"id": completion_id, "object": stream_object, "created": created,
                              "model": self.server.model_id, "choices": choices}
                if include_usage:
                    event_body["usage"] = None if not usage_marker else usage_marker
                data = json.dumps(event_body, ensure_ascii=False, separators=(",", ":"))
                with ka_lock:
                    try:
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected = False

            def _keepalive():
                # #597: an empty delta already resets the client's idle timer without
                # painting hundreds of dots in the reasoning panel during a minutes-long
                # cold prefill. COLI_VISIBLE_KEEPALIVE=1 restores the old visible "." for
                # diagnosing whether keepalives are being delivered at all.
                visible = os.environ.get("COLI_VISIBLE_KEEPALIVE") == "1"
                ping = [{"index": 0,
                         "delta": ({"reasoning_content": "." if visible else ""} if chat
                                   else {"content": ""}),
                         "logprobs": None, "finish_reason": None}]
                while not ka_stop.wait(1.0):
                    if not connected:
                        return
                    if time.time() - last_write[0] >= KA_GAP:
                        event(ping)

            def emit(text):
                choice = ({"index": 0, "delta": {"content": text}, "logprobs": None,
                           "finish_reason": None} if chat else
                          {"index": 0, "text": text, "logprobs": None, "finish_reason": None})
                event([choice])

            def emit_reasoning(text):     # thinking → reasoning_content deltas (chat only)
                event([{"index": 0, "delta": {"reasoning_content": text},
                        "logprobs": None, "finish_reason": None}])

            splitter = (InklingStreamSplit(emit, emit_reasoning if chat else None)
                        if ARCH == "inkling" else None)
            # #597 item 4: GLM (chat) streams reasoning then </think> then the answer. Split the
            # reasoning into reasoning_content deltas instead of leaking it — and the raw </think> —
            # into visible content or the tool-call buffer.
            glm_think = chat and ARCH not in ("inkling", "deepseek_v4")

            def start_stream(_accept_info=None):
                # #597 item 6: commit the streaming 200 (and start the keepalive) exactly once,
                # only after the engine ACCEPTs the prompt. Idempotent: generate() also calls this
                # on the first DATA/DONE so an older engine with no ACCEPT frame still streams.
                nonlocal connected
                if stream_started[0]:
                    return
                stream_started[0] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                # An SSE body has neither Content-Length nor chunked framing, so end-of-message
                # IS the close -- HTTP/1.1 requires us to say so, or the client waits for a
                # length that never comes and then tries to reuse a socket we are about to drop.
                # Set close_connection HERE, not after the last event: if generation raises once
                # the 200 is out, the connection must still not be offered for reuse (#597 item 3).
                self.send_header("Connection", "close")
                self.close_connection = True
                self.send_header("x-request-id", request_id)
                for name, value in queue_headers.items(): self.send_header(name, value)
                self.send_cors_headers()
                self.end_headers()
                connected = True
                last_write[0] = time.time()
                if chat:
                    event([{"index": 0, "delta": {"role": "assistant", "content": ""},
                            "logprobs": None, "finish_reason": None}])
                ka_thread[0] = threading.Thread(target=_keepalive, daemon=True)
                ka_thread[0].start()
            if chat and tools:
                # Suppress tool-call markers from the streamed content and parse the authoritative
                # calls from the FULL reply after generation. Hold back a marker-length tail so a
                # marker split across engine chunks is still caught.
                # DeepSeek V4 uses DSML markers; GLM uses <tool_call>...</tool_call>
                if ARCH == "deepseek_v4":
                    tool_marker = DSML_TOOL_CALLS_OPEN
                else:
                    tool_marker = BOX_START
                sp = {"buf": "", "tool": False}
                hold = len(tool_marker) - 1
                raw = []
                def feed_content(chunk):               # answer text only (post-</think>)
                    raw.append(chunk)
                    if sp["tool"]:
                        return
                    sp["buf"] += chunk
                    cut = sp["buf"].find(tool_marker)
                    if cut >= 0:
                        if cut:
                            emit(sp["buf"][:cut])
                        sp["buf"] = ""
                        sp["tool"] = True
                        return
                    flush = max(0, len(sp["buf"]) - hold)
                    if flush:
                        emit(sp["buf"][:flush])
                        sp["buf"] = sp["buf"][flush:]
                # #597: keep GLM reasoning out of the tool-call buffer — a think splitter sends it
                # to reasoning_content and passes only the answer text on to feed_content/parser.
                think = (ThinkingStreamSplit(emit_reasoning, feed_content,
                                             initial_thinking=enable_thinking)
                         if glm_think else None)
                def emit_tools(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (think.feed if think else feed_content)(chunk)
                stop_filter = StopFilter(stop_sequences, emit_tools, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    lambda: not connected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream, **({"audio": audio} if audio else {}))
                stop_filter.finish()
                if think:
                    think.finish()
                if not sp["tool"] and sp["buf"]:
                    emit(sp["buf"])                     # no tool call happened: flush held tail
                # Use v4 parser for DeepSeek V4, GLM parser for others
                if ARCH == "deepseek_v4":
                    _content, calls = parse_tool_calls_v4("".join(raw), tools)
                else:
                    _content, calls = parse_tool_calls("".join(raw), tools)
                for i, tc in enumerate(calls):
                    event([{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"],
                             "type": "function", "function": {"name": tc["function"]["name"],
                             "arguments": tc["function"]["arguments"]}}]},
                            "logprobs": None, "finish_reason": None}])
                finish = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            else:
                if splitter is not None:                   # inkling content/marker splitter
                    content_split = splitter
                elif glm_think:                            # GLM <think> reasoning → reasoning_content
                    content_split = ThinkingStreamSplit(emit_reasoning, emit,
                                                        initial_thinking=enable_thinking)
                else:
                    content_split = None
                def emit_plain(chunk):
                    if dbg_echo:
                        sys.stderr.write(chunk); sys.stderr.flush()
                    (content_split.feed if content_split else emit)(chunk)
                stop_filter = StopFilter(stop_sequences, emit_plain, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    lambda: not connected, grammar=grammar, stopped=stop_filter.stopped,
                    on_accept=start_stream, **({"audio": audio} if audio else {}))
                stop_filter.finish()
                if content_split:
                    content_split.close()
                finish = "length" if stats["length_limited"] else "stop"
            # generate() returned, so the prompt was ACCEPTed and start_stream() ran; guard anyway.
            start_stream()
            ka_stop.set()                          # generation done: stop the keepalive pump
            if ka_thread[0] is not None:
                ka_thread[0].join(timeout=2)
            final_choice = ({"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish}
                            if chat else {"index": 0, "text": "", "logprobs": None,
                                          "finish_reason": finish})
            event([final_choice])
            if include_usage:
                event([], self.usage(stats))
            if connected:
                with ka_lock:                          # (#B9) share the pump's lock so [DONE] can't interleave a keepalive write
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass
            # close_connection was already set when the 200 was committed (#597 item 3).

    def client_disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            return self.connection.recv(1, flags) == b""
        except (OSError, ValueError):
            return True

    @staticmethod
    def usage(stats):
        prompt = stats["prompt_tokens"]
        completion = stats["completion_tokens"]
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": prompt + completion}

    def chat_completion(self, body, request_id):
        reasoning_effort = body.get("reasoning_effort")
        efforts = (None, "none", "minimal", "low", "medium", "high", "xhigh")
        if reasoning_effort not in efforts:
            raise APIError(400, "`reasoning_effort` must be none, minimal, low, medium, high, or xhigh.",
                           "reasoning_effort")
        # COLI_THINK=1 makes thinking the default when the client sends NEITHER reasoning_effort
        # nor enable_thinking (a global switch, like the old server's --think). An explicit
        # client value always wins. Default off => exact OpenAI-standard behavior.
        if (reasoning_effort is None and "enable_thinking" not in body
                and os.environ.get("COLI_THINK", "0") == "1"):
            reasoning_effort = "high"
        enable_thinking = body.get("enable_thinking", reasoning_effort not in (None, "none"))
        if not isinstance(enable_thinking, bool):
            raise APIError(400, "`enable_thinking` must be a boolean.", "enable_thinking")
        tools = body.get("tools") or body.get("functions") or None
        tool_choice = body.get("tool_choice")
        renderer = (render_chat_inkling if ARCH == "inkling" else
                    render_chat_kimi if ARCH == "kimi" else
                    render_chat_v4 if ARCH == "deepseek_v4" else render_chat)
        audio_clips = [] if ARCH == "inkling" else None
        if audio_clips is not None:
            prompt = renderer(body.get("messages"), enable_thinking, reasoning_effort, tools,
                              tool_choice, audio_out=audio_clips)
        else:
            prompt = renderer(body.get("messages"), enable_thinking, reasoning_effort, tools,
                              tool_choice)
        self.generation(body, prompt, request_id, True, tools, tool_choice,
                        enable_thinking=enable_thinking,
                        audio=b"".join(audio_clips) if audio_clips else None)

    # ---- Anthropic /v1/messages (#343) ----------------------------------------------------
    ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}

    def anthropic_messages(self, body, request_id):
        for unsupported, why in (("stop_sequences", "custom stop sequences"),
                                 ("top_k", "top-k sampling")):
            if body.get(unsupported) not in (None, [], ""):
                raise APIError(400, f"Colibri does not support `{unsupported}` ({why}) yet.",
                               unsupported, "unsupported_value")
        messages = anthropic_to_openai(body)
        tools, tool_choice = anthropic_tools(body)
        thinking = body.get("thinking")
        if thinking is not None and not isinstance(thinking, dict):
            raise APIError(400, "`thinking` must be an object.", "thinking")
        enable_thinking = bool(thinking and thinking.get("type") == "enabled")
        if not enable_thinking and thinking is None and os.environ.get("COLI_THINK", "0") == "1":
            enable_thinking = True
        if body.get("max_tokens") is None:
            raise APIError(400, "`max_tokens` is required.", "max_tokens")
        # Reuse the OpenAI path's own validation by handing it an equivalent body.
        translated = {"messages": messages, "max_tokens": body.get("max_tokens"),
                      "temperature": body.get("temperature"), "top_p": body.get("top_p"),
                      "stream": body.get("stream", False), "cache_slot": body.get("cache_slot")}
        if tools:
            translated["tools"] = tools
        if tool_choice is not None:
            translated["tool_choice"] = tool_choice
        if tool_choice == "none":
            tools = None
        prompt = render_chat(messages, enable_thinking, "high" if enable_thinking else None,
                             tools, tool_choice)
        self.anthropic_generation(translated, prompt, request_id, tools, enable_thinking)

    def anthropic_generation(self, body, prompt, request_id, tools, enable_thinking):
        maximum, temperature, top_p, grammar, _stop_sequences = generation_options(
            body, self.server.max_tokens)
        # Same policy as /v1/chat/completions: `body` is the translated OpenAI-shaped
        # request, and anthropic_messages() has already refused a client `stop_sequences`,
        # so this resolves to the implicit GLM role boundaries.
        stop_sequences, ignore_leading_stop = stop_policy(body, True)
        cache_slot = body.get("cache_slot")
        if (cache_slot is not None and
                (isinstance(cache_slot, bool) or not isinstance(cache_slot, int) or
                 not 0 <= cache_slot < self.server.kv_slots)):
            raise APIError(400, f"`cache_slot` must be an integer between 0 and {self.server.kv_slots - 1}.",
                           "cache_slot")
        if cache_slot is None and self.server.kv_slots > 1:
            # #634: pin each conversation to a stable KV slot so multi-turn reuses its
            # cached prefix instead of re-prefilling. Only when the request carries a
            # conversation; raw /v1/completions keeps the scheduler's free-slot pick.
            conversation = body.get("messages")
            if isinstance(conversation, list) and conversation:
                cache_slot = conversation_cache_slot(conversation, self.server.kv_slots)
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(400, "`stream` must be a boolean.", "stream")
        message_id = "msg_" + uuid.uuid4().hex[:24]

        def blocks_and_stop(text, stats):
            """Split a finished reply into Anthropic content blocks + stop_reason."""
            content = []
            if enable_thinking:
                reasoning, text = split_thinking_reply(text)
                content.append({"type": "thinking", "thinking": reasoning,
                                "signature": ANTHROPIC_LOCAL_SIGNATURE})
            calls = []
            if tools:
                text, calls = parse_tool_calls(text, tools)
            if text:
                content.append({"type": "text", "text": text})
            for call in calls:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content.append({"type": "tool_use", "id": call["id"],
                                "name": function["name"], "input": arguments})
            reason = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            return content, self.ANTHROPIC_STOP[reason]

        with self.server.scheduler.admit(self.client_disconnected, cache_slot) as admission:
            queue_wait, cache_slot = admission
            queue_headers = {"x-colibri-queue-wait-ms": str(round(queue_wait * 1000))}
            if not stream:
                output = []
                stop_filter = StopFilter(stop_sequences, output.append, ignore_leading_stop)
                stats = self.server.engine.generate(
                    prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                    self.client_disconnected, grammar=grammar, stopped=stop_filter.stopped)
                stop_filter.finish()
                content, stop_reason = blocks_and_stop("".join(output), stats)
                self.send_json(200, {
                    "id": message_id, "type": "message", "role": "assistant",
                    "model": self.server.model_id, "content": content,
                    "stop_reason": stop_reason, "stop_sequence": None,
                    "usage": {"input_tokens": stats["prompt_tokens"],
                              "output_tokens": stats["completion_tokens"]}},
                    request_id, queue_headers)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")   # see the OpenAI path: SSE is close-framed
            self.close_connection = True
            self.send_header("x-request-id", request_id)
            for name, value in queue_headers.items():
                self.send_header(name, value)
            self.send_cors_headers()
            self.end_headers()
            connected = [True]
            write_lock = threading.Lock()
            last_write = [time.time()]
            ka_stop = threading.Event()

            def send_event(name, payload):
                if not connected[0]:
                    return
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                with write_lock:
                    try:
                        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode())
                        self.wfile.flush()
                        last_write[0] = time.time()
                    except OSError:
                        connected[0] = False

            # Anthropic has a first-class keepalive event, so the cold prefill (minutes) does
            # not need the OpenAI path's reasoning-delta trick: `ping` is in the protocol.
            def keepalive():
                while not ka_stop.wait(1.0):
                    if not connected[0]:
                        return
                    if time.time() - last_write[0] >= 10.0:
                        send_event("ping", {"type": "ping"})

            send_event("message_start", {"type": "message_start", "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": self.server.model_id, "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}})
            text_index = 1 if enable_thinking else 0
            stream_state = {"thinking_closed": not enable_thinking,
                            "text_started": not enable_thinking}
            if enable_thinking:
                send_event("content_block_start", {"type": "content_block_start", "index": 0,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            else:
                send_event("content_block_start", {"type": "content_block_start", "index": 0,
                                                   "content_block": {"type": "text", "text": ""}})
            ka_thread = threading.Thread(target=keepalive, daemon=True)
            ka_thread.start()

            raw = []
            state = {"buf": "", "in_tool": False}
            hold = len(BOX_START) - 1

            def emit_text(chunk):
                if not chunk:
                    return
                state["buf"] += chunk
                cut = state["buf"].find(BOX_START)
                if cut >= 0:
                    if cut:
                        send_event("content_block_start", {"type": "content_block_start",
                            "index": text_index, "content_block": {"type": "text", "text": ""}})
                        stream_state["text_started"] = True
                        send_event("content_block_delta", {"type": "content_block_delta",
                            "index": text_index, "delta": {"type": "text_delta", "text": state["buf"][:cut]}})
                    state["buf"] = ""
                    state["in_tool"] = True
                    return
                flush = max(0, len(state["buf"]) - hold)
                if flush:
                    if not stream_state["text_started"]:
                        send_event("content_block_start", {"type": "content_block_start",
                            "index": text_index, "content_block": {"type": "text", "text": ""}})
                        stream_state["text_started"] = True
                    send_event("content_block_delta", {"type": "content_block_delta",
                        "index": text_index, "delta": {"type": "text_delta", "text": state["buf"][:flush]}})
                    state["buf"] = state["buf"][flush:]

            def emit_thinking(chunk):
                if not chunk:
                    return
                send_event("content_block_delta", {"type": "content_block_delta",
                    "index": 0, "delta": {"type": "thinking_delta", "thinking": chunk,
                    "signature": ANTHROPIC_LOCAL_SIGNATURE}})

            think = ThinkingStreamSplit(emit_thinking, emit_text,
                                        initial_thinking=enable_thinking)
            stop_filter = StopFilter(stop_sequences, think.feed, ignore_leading_stop)
            stats = self.server.engine.generate(
                prompt, maximum, temperature, top_p, stop_filter.feed, cache_slot,
                lambda: not connected[0], grammar=grammar, stopped=stop_filter.stopped)
            stop_filter.finish()
            think.finish()
            if not state["in_tool"] and state["buf"]:
                emit_text(state["buf"])

            # Parse tool calls from raw buffer
            _content, calls = parse_tool_calls("".join(raw), tools) if raw else ("", [])
            # Also try parsing from the full output if no raw collected
            if not raw:
                # Reconstruct from state
                pass

            # For Anthropic, emit tool_use blocks
            for i, call in enumerate(calls):
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_index = text_index + i
                send_event("content_block_start", {"type": "content_block_start",
                    "index": tool_index,
                    "content_block": {"type": "tool_use", "id": call["id"],
                                      "name": function["name"], "input": {}}})
                send_event("content_block_delta", {"type": "content_block_delta",
                    "index": tool_index, "delta": {"type": "input_json_delta",
                    "partial_json": function["arguments"]}})

            stop_reason = "tool_calls" if calls else ("length" if stats["length_limited"] else "stop")
            send_event("message_delta", {"type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": stats["prompt_tokens"],
                          "output_tokens": stats["completion_tokens"]}})
            send_event("message_stop", {"type": "message_stop"})
            ka_stop.set()
            if ka_thread:
                ka_thread.join(timeout=2)

    def completion(self, body, request_id):
        """Raw /v1/completions: single prompt, no chat template."""
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise APIError(400, "`prompt` must be a string.", "prompt")
        self.generation(body, prompt, request_id, False)

    def _committed(self):
        return getattr(self, '_committed', False)


def serve(model, host="127.0.0.1", port=8000, model_id="glm-5.2-colibri", api_key=None,
          cap=None, max_tokens=1024, engine=None, env=None, cors_origins=None,
          max_queue=8, queue_timeout=300, kv_slots=1, allowed_hosts=()):
    """Start the HTTP server (legacy public API, called by the coli CLI).

    Parameters not supported by the new backend (api_key, cors_origins,
    allowed_hosts) are accepted for backward compatibility but ignored.
    """
    if engine is None:
        engine = default_engine()

    class Args:
        pass

    args = Args()
    args.engine = engine
    args.model = model
    args.host = host
    args.port = port
    args.model_id = model_id
    args.max_tokens = max_tokens
    args.max_queue = max_queue
    args.queue_timeout = queue_timeout
    args.kv_slots = kv_slots
    args.ctx = 8192
    args.max_batch = 2048
    args.gpu_layer = 0
    args.tensor_split = None
    args.mlock = False
    args.flash_attn = False
    args.threads = 0
    args.threads_batch = 0
    args.verbose = False
    args.cap = cap
    args.env = env
    args.arch = ARCH if ARCH != "auto" else model_arch(model)

    _serve(args)


def _serve(args):
    """Start the HTTP server."""
    engine = Engine(args.engine, args.model, args.ctx, args.max_tokens, args.max_batch, args.kv_slots,
                    args.gpu_layer, args.tensor_split, args.mlock, args.flash_attn,
                    args.threads, args.threads_batch, args.verbose, cap=getattr(args, "cap", None),
                    model_arch=args.arch, env=getattr(args, "env", None))
    engine.start()

    class Server(ThreadingHTTPServer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.engine = engine
            self.model_id = args.model_id
            self.max_tokens = args.max_tokens
            self.kv_slots = args.kv_slots
            self.scheduler = GenerationScheduler(args.max_queue, args.queue_timeout, args.kv_slots)
            self._profile_data = []

        def _profile(self):
            return {"turns": self._profile_data[-PROFILE_TURNS:],
                    "scheduler": self.scheduler.snapshot()}

    server = Server(("" if args.host == "0.0.0.0" else args.host, args.port), ColibriHandler)

    def shutdown(signum, frame):
        sys.stderr.write(f"\nShutting down on signal {signum}...\n")
        sys.stderr.flush()
        server.shutdown()
        engine.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    sys.stderr.write(f"Colibri OpenAI API server listening on {args.host}:{args.port}\n")
    sys.stderr.write(f"Engine: {args.engine}, Model: {args.model_id}, "
                     f"Arch: {args.arch}, Max tokens: {args.max_tokens}\n")
    sys.stderr.flush()

    try:
        server.serve_forever()
    finally:
        engine.close()


def main():
    global ARCH
    parser = argparse.ArgumentParser(description="Colibri OpenAI-compatible API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--model", type=str, required=True, help="Model file path")
    parser.add_argument("--ctx", type=int, default=8192, help="Context size (default: 8192)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument("--max-batch", type=int, default=2048, help="Max batch size (default: 2048)")
    parser.add_argument("--kv-slots", type=int, default=1, help="Number of KV slots (default: 1)")
    parser.add_argument("--gpu-layer", type=int, default=9999, help="GPU layers (default: all)")
    parser.add_argument("--tensor-split", type=str, default=None, help="Tensor split ratio")
    parser.add_argument("--mlock", action="store_true", help="Lock memory")
    parser.add_argument("--flash-attn", action="store_true", help="Flash attention")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads")
    parser.add_argument("--threads-batch", type=int, default=0, help="Batch threads")
    parser.add_argument("--verbose", action="store_true", help="Verbose engine output")
    parser.add_argument("--max-queue", type=int, default=8, help="Max queue size (default: 8)")
    parser.add_argument("--queue-timeout", type=int, default=300, help="Queue timeout (default: 300s)")
    parser.add_argument("--arch", choices=("auto", "glm", "inkling", "kimi", "deepseek_v4"), default="auto",
                        help="Model architecture (default: auto-detect)")
    parser.add_argument("--engine", default=None, help="Path to the engine binary")
    parser.add_argument("--model-id", default=None, help="Model ID for the API")
    args = parser.parse_args()

    ARCH = args.arch
    if ARCH == "auto":
        ARCH = model_arch(args.model)
    args.arch = ARCH
    if args.model_id is None:
        args.model_id = ("inkling-colibri" if ARCH == "inkling" else
                         "kimi-k3-colibri" if ARCH == "kimi" else
                         "deepseek-v4-colibri" if ARCH == "deepseek_v4" else
                         "glm-colibri")
    if args.engine is None:
        args.engine = default_engine()
    _serve(args)


if __name__ == "__main__":
    main()
