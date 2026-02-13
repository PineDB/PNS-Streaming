#!/usr/bin/env python3
"""
stream_bvh.py — Connect to Noitom Axis Studio's BVH Broadcasting via TCP,
parse the motion-capture data, and live-plot a 3D skeleton.

Usage:
    python stream_bvh.py [--ip IP] [--port PORT]

Defaults to 127.0.0.1:7007 (Adv BVH Broadcasting).
"""

from __future__ import annotations

import argparse
import gzip
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# ---------------------------------------------------------------------------
# 1.  Axis Studio skeleton definition
#     59 joints, depth-first BVH order.
#     Offsets taken from the standard Axis Studio basic_hierarchy.bvh.
# ---------------------------------------------------------------------------

# (name, parent_index, offset_x, offset_y, offset_z)
# parent_index = -1 means root (no parent)
JOINT_DEFS = [
    # idx  0 — root
    ("Hips",               -1,    0.000,  97.120,   0.000),
    # Right leg
    ("RightUpLeg",          0,  -11.000,   0.000,   0.000),  # 1
    ("RightLeg",            1,    0.000, -45.062,   0.000),  # 2
    ("RightFoot",           2,    0.000, -42.058,   0.000),  # 3
    # Left leg
    ("LeftUpLeg",           0,   11.000,   0.000,   0.000),  # 4
    ("LeftLeg",             4,    0.000, -45.062,   0.000),  # 5
    ("LeftFoot",            5,    0.000, -42.058,   0.000),  # 6
    # Spine chain
    ("Spine",               0,    0.000,   8.120,   0.000),  # 7
    ("Spine1",              7,    0.000,  17.980,   0.000),  # 8
    ("Spine2",              8,    0.000,  12.760,   0.000),  # 9
    # Neck / head
    ("Neck",                9,    0.000,  19.140,   0.000),  # 10
    ("Neck1",              10,    0.000,   4.250,   0.000),  # 11
    ("Head",               11,    0.000,   4.250,   0.000),  # 12
    # Right arm
    ("RightShoulder",       9,   -2.900,  13.340,   0.000),  # 13
    ("RightArm",           13,  -16.100,   0.000,   0.000),  # 14
    ("RightForeArm",       14,  -28.000,   0.000,   0.000),  # 15
    ("RightHand",          15,  -26.000,   0.000,   0.000),  # 16
    # Right hand fingers
    ("RightHandThumb1",    16,   -1.937,  -0.484,   2.518),  # 17
    ("RightHandThumb2",    17,   -3.872,   0.000,   0.000),  # 18
    ("RightHandThumb3",    18,   -2.690,   0.000,   0.000),  # 19
    ("RightInHandIndex",   16,   -3.389,   0.535,   2.080),  # 20
    ("RightHandIndex1",    20,   -5.485,  -0.096,   1.051),  # 21
    ("RightHandIndex2",    21,   -3.806,   0.000,   0.000),  # 22
    ("RightHandIndex3",    22,   -2.158,   0.000,   0.000),  # 23
    ("RightInHandMiddle",  16,   -3.556,   0.544,   0.796),  # 24
    ("RightHandMiddle1",   24,   -5.441,  -0.088,   0.330),  # 25
    ("RightHandMiddle2",   25,   -4.153,   0.000,   0.000),  # 26
    ("RightHandMiddle3",   26,   -2.603,   0.000,   0.000),  # 27
    ("RightInHandRing",    16,   -3.539,   0.566,  -0.136),  # 28
    ("RightHandRing1",     28,   -4.873,  -0.023,  -0.504),  # 29
    ("RightHandRing2",     29,   -3.619,   0.000,   0.000),  # 30
    ("RightHandRing3",     30,   -2.511,   0.000,   0.000),  # 31
    ("RightInHandPinky",   16,   -3.324,   0.494,  -1.264),  # 32
    ("RightHandPinky1",    32,   -4.354,  -0.023,  -1.147),  # 33
    ("RightHandPinky2",    33,   -2.898,   0.000,   0.000),  # 34
    ("RightHandPinky3",    34,   -1.831,   0.000,   0.000),  # 35
    # Left arm
    ("LeftShoulder",        9,    2.900,  13.340,   0.000),  # 36
    ("LeftArm",            36,   16.100,   0.000,   0.000),  # 37
    ("LeftForeArm",        37,   28.000,   0.000,   0.000),  # 38
    ("LeftHand",           38,   26.000,   0.000,   0.000),  # 39
    # Left hand fingers
    ("LeftHandThumb1",     39,    1.937,  -0.484,   2.518),  # 40
    ("LeftHandThumb2",     40,    3.872,   0.000,   0.000),  # 41
    ("LeftHandThumb3",     41,    2.690,   0.000,   0.000),  # 42
    ("LeftInHandIndex",    39,    3.389,   0.535,   2.080),  # 43
    ("LeftHandIndex1",     43,    5.485,  -0.096,   1.051),  # 44
    ("LeftHandIndex2",     44,    3.806,   0.000,   0.000),  # 45
    ("LeftHandIndex3",     45,    2.158,   0.000,   0.000),  # 46
    ("LeftInHandMiddle",   39,    3.556,   0.544,   0.796),  # 47
    ("LeftHandMiddle1",    47,    5.441,  -0.088,   0.330),  # 48
    ("LeftHandMiddle2",    48,    4.153,   0.000,   0.000),  # 49
    ("LeftHandMiddle3",    49,    2.603,   0.000,   0.000),  # 50
    ("LeftInHandRing",     39,    3.539,   0.566,  -0.136),  # 51
    ("LeftHandRing1",      51,    4.873,  -0.023,  -0.504),  # 52
    ("LeftHandRing2",      52,    3.619,   0.000,   0.000),  # 53
    ("LeftHandRing3",      53,    2.511,   0.000,   0.000),  # 54
    ("LeftInHandPinky",    39,    3.324,   0.494,  -1.264),  # 55
    ("LeftHandPinky1",     55,    4.354,  -0.023,  -1.147),  # 56
    ("LeftHandPinky2",     56,    2.898,   0.000,   0.000),  # 57
    ("LeftHandPinky3",     57,    1.831,   0.000,   0.000),  # 58
]

NUM_JOINTS = len(JOINT_DEFS)  # 59

JOINT_NAMES = [j[0] for j in JOINT_DEFS]
JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Bones to draw — every parent→child connection in the skeleton.
BODY_BONES = [(j[1], i) for i, j in enumerate(JOINT_DEFS) if j[1] >= 0]

# Colour per bone group for the plot
BONE_COLOURS = {
    "right_leg":     "#e74c3c",   # red
    "left_leg":      "#2ecc71",   # green
    "spine":         "#f1c40f",   # yellow
    "head":          "#9b59b6",   # purple
    "right_arm":     "#e67e22",   # orange
    "left_arm":      "#3498db",   # blue
    "right_fingers": "#d35400",   # dark orange
    "left_fingers":  "#2980b9",   # dark blue
}

def _bone_colour(parent_idx: int, child_idx: int) -> str:
    """Pick a colour based on which body part the bone belongs to."""
    names = {JOINT_NAMES[parent_idx], JOINT_NAMES[child_idx]}
    for n in names:
        if "Right" in n and ("Leg" in n or "Foot" in n):
            return BONE_COLOURS["right_leg"]
        if "Left" in n and ("Leg" in n or "Foot" in n):
            return BONE_COLOURS["left_leg"]
        if "Right" in n and ("Thumb" in n or "Index" in n or "Middle" in n
                             or "Ring" in n or "Pinky" in n or "InHand" in n):
            return BONE_COLOURS["right_fingers"]
        if "Left" in n and ("Thumb" in n or "Index" in n or "Middle" in n
                            or "Ring" in n or "Pinky" in n or "InHand" in n):
            return BONE_COLOURS["left_fingers"]
        if "Right" in n and ("Shoulder" in n or "Arm" in n or "Hand" in n):
            return BONE_COLOURS["right_arm"]
        if "Left" in n and ("Shoulder" in n or "Arm" in n or "Hand" in n):
            return BONE_COLOURS["left_arm"]
        if "Head" in n or "Neck" in n:
            return BONE_COLOURS["head"]
    return BONE_COLOURS["spine"]


# ---------------------------------------------------------------------------
# 2.  Forward kinematics
# ---------------------------------------------------------------------------

def _rotation_matrix_yxz(y_deg: float, x_deg: float, z_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for YXZ intrinsic Euler angles (degrees)."""
    yr, xr, zr = math.radians(y_deg), math.radians(x_deg), math.radians(z_deg)
    cy, sy = math.cos(yr), math.sin(yr)
    cx, sx = math.cos(xr), math.sin(xr)
    cz, sz = math.cos(zr), math.sin(zr)

    # R = Ry * Rx * Rz
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def forward_kinematics(channel_data: list[float]) -> np.ndarray:
    """
    Compute world positions for every joint.

    Parameters
    ----------
    channel_data : list of 180 floats
        Root: Xpos Ypos Zpos Yrot Xrot Zrot
        Each subsequent joint: Yrot Xrot Zrot

    Returns
    -------
    positions : ndarray of shape (59, 3)
    """
    positions = np.zeros((NUM_JOINTS, 3))
    world_transforms = [np.eye(4) for _ in range(NUM_JOINTS)]

    # Build channel index: root gets 6, rest get 3 each
    ch_idx = 0

    for j_idx, (name, parent_idx, off_x, off_y, off_z) in enumerate(JOINT_DEFS):
        # ---- parent transform ----
        if parent_idx >= 0:
            parent_T = world_transforms[parent_idx]
        else:
            parent_T = np.eye(4)

        # ---- local transform ----
        T_local = np.eye(4)

        if j_idx == 0:
            # Root: read position + rotation
            px, py, pz = channel_data[ch_idx], channel_data[ch_idx + 1], channel_data[ch_idx + 2]
            ry, rx, rz = channel_data[ch_idx + 3], channel_data[ch_idx + 4], channel_data[ch_idx + 5]
            ch_idx += 6
            # Translate to root position (ignore rest-pose offset; the streamed
            # position already represents the world position of the hips).
            T_local[:3, 3] = [px, py, pz]
            T_local[:3, :3] = _rotation_matrix_yxz(ry, rx, rz)
        else:
            # Other joints: translate by bone offset, then rotate
            ry, rx, rz = channel_data[ch_idx], channel_data[ch_idx + 1], channel_data[ch_idx + 2]
            ch_idx += 3
            T_local[:3, 3] = [off_x, off_y, off_z]
            T_local[:3, :3] = _rotation_matrix_yxz(ry, rx, rz)

        world_T = parent_T @ T_local
        world_transforms[j_idx] = world_T
        positions[j_idx] = world_T[:3, 3]

    return positions


# ---------------------------------------------------------------------------
# 3.  Protobuf wire-format helpers  (no external dependency needed)
# ---------------------------------------------------------------------------

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read a protobuf varint starting at *pos*. Return (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("Truncated varint")


def _try_parse_protobuf(data: bytes):
    """
    Attempt to decode *data* as a protobuf message (no schema).

    Returns a list of (field_number, wire_type, raw_value) tuples,
    or None if the data doesn't look like valid protobuf.
    """
    fields = []
    pos = 0
    while pos < len(data):
        try:
            tag, pos = _read_varint(data, pos)
        except ValueError:
            return None
        field_num = tag >> 3
        wire_type = tag & 0x7
        if field_num == 0 or field_num > 536_870_911:
            return None
        if wire_type == 0:          # varint
            try:
                val, pos = _read_varint(data, pos)
            except ValueError:
                return None
            fields.append((field_num, 0, val))
        elif wire_type == 1:        # 64-bit
            if pos + 8 > len(data):
                return None
            fields.append((field_num, 1, data[pos:pos + 8]))
            pos += 8
        elif wire_type == 2:        # length-delimited
            try:
                length, pos = _read_varint(data, pos)
            except ValueError:
                return None
            if length < 0 or pos + length > len(data):
                return None
            fields.append((field_num, 2, data[pos:pos + length]))
            pos += length
        elif wire_type == 5:        # 32-bit (float / fixed32)
            if pos + 4 > len(data):
                return None
            fields.append((field_num, 5, data[pos:pos + 4]))
            pos += 4
        else:
            return None             # wire types 3,4 are deprecated / unknown
    return fields


def _collect_floats_protobuf(data: bytes, depth: int = 0) -> list[float]:
    """
    Recursively walk protobuf wire format and collect every 32-bit float
    value (wire type 5) and every packed-float array (length-delimited with
    length divisible by 4 that yields sensible numbers).
    """
    fields = _try_parse_protobuf(data)
    if fields is None:
        return []

    floats: list[float] = []
    for _fnum, wtype, raw in fields:
        if wtype == 5:
            # Individual 32-bit float
            floats.append(struct.unpack("<f", raw)[0])
        elif wtype == 2 and len(raw) >= 4:
            # Try as nested protobuf first
            nested = _collect_floats_protobuf(raw, depth + 1)
            if nested:
                floats.extend(nested)
            elif len(raw) % 4 == 0:
                # Try as packed repeated float
                packed = list(struct.unpack(f"<{len(raw) // 4}f", raw))
                if all(not math.isnan(v) and not math.isinf(v) for v in packed):
                    floats.extend(packed)
    return floats


def _dump_protobuf(data: bytes, indent: int = 0) -> str:
    """Pretty-print raw protobuf structure (for debugging)."""
    fields = _try_parse_protobuf(data)
    if fields is None:
        return f"{'  ' * indent}(not valid protobuf, {len(data)} bytes)"

    lines = []
    for fnum, wtype, raw in fields:
        prefix = "  " * indent
        if wtype == 0:
            lines.append(f"{prefix}field {fnum} (varint): {raw}")
        elif wtype == 1:
            dval = struct.unpack("<d", raw)[0]
            lines.append(f"{prefix}field {fnum} (64-bit): {dval}")
        elif wtype == 5:
            fval = struct.unpack("<f", raw)[0]
            lines.append(f"{prefix}field {fnum} (float32): {fval}")
        elif wtype == 2:
            # Try as nested message
            nested = _try_parse_protobuf(raw)
            if nested is not None and len(nested) > 0:
                lines.append(f"{prefix}field {fnum} (message, {len(raw)} bytes):")
                lines.append(_dump_protobuf(raw, indent + 1))
            else:
                # Try as string
                try:
                    s = raw.decode("utf-8")
                    if all(32 <= ord(c) < 127 for c in s):
                        lines.append(f"{prefix}field {fnum} (string): {s!r}")
                        continue
                except UnicodeDecodeError:
                    pass
                # Show as hex
                preview = raw[:40].hex(" ")
                lines.append(f"{prefix}field {fnum} (bytes, {len(raw)}B): {preview}{'…' if len(raw) > 40 else ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4.  TCP receiver thread
# ---------------------------------------------------------------------------

# Protocol markers
CCCC_HEADER = b"CCCC"                  # Adv BVH Broadcasting (gzip-compressed)
BINARY_HEADER_TOKEN = 0xDDFF           # Standard BVH Broadcasting (binary)
GZIP_MAGIC = b"\x1f\x8b"              # gzip magic bytes

# Format enum
FMT_UNKNOWN = 0
FMT_CCCC_GZIP = 1   # Adv BVH: CCCC + uint32 length + gzip payload
FMT_BINARY = 2      # Standard BVH binary (0xDDFF header)
FMT_TEXT = 3         # Plain text frames delimited by ||


class BVHReceiver:
    """Receive BVH frames from Axis Studio over TCP in a background thread."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 7007):
        self.ip = ip
        self.port = port
        self.frames: deque[list[float]] = deque(maxlen=2)  # latest frame(s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.frame_count = 0
        self.fps = 0.0
        self._fmt = FMT_UNKNOWN
        self._debug_packets = 0
        self._debug_frame_warns = 0

    # ---- public API ----

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def latest_frame(self) -> list[float] | None:
        """Return the most recent parsed frame (or None)."""
        try:
            return self.frames[-1]
        except IndexError:
            return None

    # ---- internal ----

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._connect_and_receive()
            except (ConnectionRefusedError, ConnectionResetError, OSError) as exc:
                self.connected = False
                print(f"[receiver] Connection error: {exc}  — retrying in 2 s …")
                time.sleep(2)

    def _connect_and_receive(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            print(f"[receiver] Connecting to {self.ip}:{self.port} …")
            sock.connect((self.ip, self.port))
            sock.settimeout(1)  # allow periodic stop-event checks
            self.connected = True
            self._fmt = FMT_UNKNOWN
            self._debug_packets = 0
            print(f"[receiver] Connected!")

            buf = b""
            fps_t0 = time.time()
            fps_count = 0

            while not self._stop_event.is_set():
                try:
                    chunk = sock.recv(16384)
                except socket.timeout:
                    continue
                if not chunk:
                    print("[receiver] Server closed connection.")
                    self.connected = False
                    return

                buf += chunk

                # Auto-detect format on first data
                if self._fmt == FMT_UNKNOWN and len(buf) >= 8:
                    if buf[:4] == CCCC_HEADER:
                        self._fmt = FMT_CCCC_GZIP
                        print("[receiver] Detected Adv BVH format (CCCC + gzip).")
                    elif struct.unpack_from("<H", buf, 0)[0] == BINARY_HEADER_TOKEN:
                        self._fmt = FMT_BINARY
                        print("[receiver] Detected standard BVH binary format.")
                    else:
                        self._fmt = FMT_TEXT
                        print("[receiver] Detected text format.")

                if self._fmt == FMT_CCCC_GZIP:
                    buf = self._parse_cccc_buffer(buf)
                elif self._fmt == FMT_BINARY:
                    buf = self._parse_binary_buffer(buf)
                else:
                    buf = self._parse_text_buffer(buf)

                # FPS bookkeeping
                fps_count += 1
                elapsed = time.time() - fps_t0
                if elapsed >= 1.0:
                    self.fps = fps_count / elapsed
                    fps_count = 0
                    fps_t0 = time.time()

    # ------------------------------------------------------------------ #
    #  Adv BVH protocol:  [CCCC][uint32 LE length][gzip payload] ...     #
    # ------------------------------------------------------------------ #

    def _parse_cccc_buffer(self, buf: bytes) -> bytes:
        HEADER_SIZE = 8  # 4 bytes "CCCC" + 4 bytes length
        while len(buf) >= HEADER_SIZE:
            # Find next CCCC marker (re-sync if needed)
            marker_pos = buf.find(CCCC_HEADER)
            if marker_pos < 0:
                return b""
            if marker_pos > 0:
                buf = buf[marker_pos:]

            if len(buf) < HEADER_SIZE:
                break

            payload_len = struct.unpack_from("<I", buf, 4)[0]
            packet_len = HEADER_SIZE + payload_len

            if len(buf) < packet_len:
                break  # wait for full packet

            payload = buf[HEADER_SIZE:packet_len]
            buf = buf[packet_len:]

            # Decompress gzip
            try:
                decompressed = gzip.decompress(payload)
            except Exception as exc:
                if self._debug_frame_warns < 5:
                    self._debug_frame_warns += 1
                    print(f"[receiver] gzip decompress error: {exc}")
                continue

            # Debug: print protobuf structure of first few packets
            if self._debug_packets < 3:
                self._debug_packets += 1
                print(f"\n[receiver] Decompressed packet #{self._debug_packets} "
                      f"({len(decompressed)} bytes):")
                print(_dump_protobuf(decompressed, indent=1))
                floats = _collect_floats_protobuf(decompressed)
                print(f"  → Extracted {len(floats)} float(s): "
                      f"{floats[:15]}{'…' if len(floats) > 15 else ''}")

            self._handle_decompressed_packet(decompressed)

        return buf

    def _handle_decompressed_packet(self, data: bytes):
        """Parse the decompressed content of a CCCC packet (protobuf)."""
        # 1. Try protobuf → extract floats
        floats = _collect_floats_protobuf(data)
        if floats:
            self._accept_channel_values(floats)
            return

        # 2. Fallback: try as UTF-8 text (|| delimited or line-based)
        try:
            text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return

        if not text:
            return
        if text.startswith("HIERARCHY"):
            print("[receiver] Received BVH hierarchy header.")
            return
        if "||" in text:
            for seg in text.split("||"):
                seg = seg.strip()
                if seg:
                    self._handle_text_frame(seg)
            return
        for line in text.splitlines():
            line = line.strip()
            if line:
                self._handle_text_frame(line)

    # ------------------------------------------------------------------ #
    #  Text frames:  "<index> <name> <float> <float> ... "               #
    # ------------------------------------------------------------------ #

    def _parse_text_buffer(self, buf: bytes) -> bytes:
        text = buf.decode("utf-8", errors="replace")
        while "||" in text:
            frame_str, text = text.split("||", 1)
            self._handle_text_frame(frame_str.strip())
        return text.encode("utf-8")

    def _handle_text_frame(self, frame_str: str):
        tokens = frame_str.split()
        if len(tokens) < 3:
            return

        # First token: frame index,  second: avatar name,  rest: float channels.
        # Some formats may not have the index/name prefix — try to detect.
        try:
            values = [float(t) for t in tokens]
            # All tokens are floats — no index/name prefix
            self._accept_channel_values(values)
            return
        except ValueError:
            pass

        # Skip leading non-float tokens (index, avatar name, etc.)
        float_start = 0
        for i, t in enumerate(tokens):
            try:
                float(t)
                float_start = i
                break
            except ValueError:
                continue
        else:
            return  # no floats found

        try:
            values = [float(t) for t in tokens[float_start:]]
        except ValueError:
            return

        self._accept_channel_values(values)

    # ------------------------------------------------------------------ #
    #  Accept a list of float channel values (shared by all parsers)      #
    # ------------------------------------------------------------------ #

    def _accept_channel_values(self, values: list[float]):
        expected_no_disp = 6 + (NUM_JOINTS - 1) * 3  # 180
        expected_with_disp = NUM_JOINTS * 6            # 354

        if len(values) == expected_no_disp:
            self.frames.append(values)
            self.frame_count += 1
        elif len(values) == expected_with_disp:
            # Displacement enabled — convert to 180-channel format
            converted: list[float] = []
            for j in range(NUM_JOINTS):
                base = j * 6
                if j == 0:
                    converted.extend(values[base:base + 6])
                else:
                    converted.extend(values[base + 3:base + 6])
            self.frames.append(converted)
            self.frame_count += 1
        else:
            if self._debug_frame_warns < 10:
                self._debug_frame_warns += 1
                print(f"[receiver] Frame with {len(values)} channels "
                      f"(expected {expected_no_disp} or {expected_with_disp}). "
                      f"First 10: {values[:10]}")

    # ------------------------------------------------------------------ #
    #  Standard BVH binary protocol (0xDDFF header)                      #
    # ------------------------------------------------------------------ #

    def _parse_binary_buffer(self, buf: bytes) -> bytes:
        HEADER_LEN = 64
        while len(buf) >= HEADER_LEN:
            token = struct.unpack_from("<H", buf, 0)[0]
            if token != BINARY_HEADER_TOKEN:
                idx = buf.find(struct.pack("<H", BINARY_HEADER_TOKEN), 1)
                if idx < 0:
                    return b""
                buf = buf[idx:]
                continue

            data_count = struct.unpack_from("<H", buf, 6)[0]
            frame_len = HEADER_LEN + data_count * 4
            if len(buf) < frame_len:
                break

            data_section = buf[HEADER_LEN:frame_len]
            buf = buf[frame_len:]

            floats = list(struct.unpack_from(f"<{data_count}f", data_section))
            self._accept_channel_values(floats)

        return buf


# ---------------------------------------------------------------------------
# 4.  Live 3D matplotlib plot
# ---------------------------------------------------------------------------

def run_live_plot(receiver: BVHReceiver):
    """Open a matplotlib window and animate the skeleton in real time."""

    fig = plt.figure("Axis Studio — BVH Live Viewer", figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Pre-compute bone line colours and widths (fingers are thinner)
    bone_colours = [_bone_colour(p, c) for p, c in BODY_BONES]
    _FINGER_COLOURS = {BONE_COLOURS["right_fingers"], BONE_COLOURS["left_fingers"]}

    # Create line objects for each bone
    lines = []
    for colour in bone_colours:
        lw = 1.2 if colour in _FINGER_COLOURS else 2.5
        (line,) = ax.plot([], [], [], lw=lw, color=colour)
        lines.append(line)

    # Joint scatter
    scatter = ax.scatter([], [], [], s=18, c="#ecf0f1", edgecolors="#2c3e50",
                         linewidths=0.5, depthshade=True)

    title_text = ax.set_title("Waiting for data …", fontsize=11)

    # Axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # forward
    ax.set_zlabel("Y")  # up

    # Fixed axis range — large enough for a full-body skeleton moving in space.
    # Centered around the origin; the skeleton's root is typically near Y≈97.
    axis_range = [200]  # half-extent in each direction (cm), mutable for zoom
    CENTER_X, CENTER_Z, CENTER_Y = 0, 0, 100  # plot coords (X, Z_forward, Y_up)

    def _apply_axis_range():
        r = axis_range[0]
        ax.set_xlim(CENTER_X - r, CENTER_X + r)
        ax.set_ylim(CENTER_Z - r, CENTER_Z + r)
        ax.set_zlim(CENTER_Y - r, CENTER_Y + r)

    _apply_axis_range()

    # Keyboard zoom: Cmd+Plus to zoom in, Cmd+Minus to zoom out
    # (matplotlib reports Cmd as "super")
    ZOOM_FACTOR = 1.25

    def _on_key(event):
        if event.key in ("+", "=", "super+=", "super++", "ctrl+=", "ctrl++"):
            axis_range[0] = max(20, axis_range[0] / ZOOM_FACTOR)
            _apply_axis_range()
            fig.canvas.draw_idle()
        elif event.key in ("-", "super+-", "super+-", "ctrl+-"):
            axis_range[0] = min(2000, axis_range[0] * ZOOM_FACTOR)
            _apply_axis_range()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    def update(frame_number):
        data = receiver.latest_frame()
        if data is None:
            status = "Connecting …" if not receiver.connected else "Waiting for frames …"
            title_text.set_text(status)
            return lines + [scatter]

        positions = forward_kinematics(data)

        # We plot X, Z, Y  (swap Y↔Z so "up" is the matplotlib Z-axis)
        xs = positions[:, 0]
        ys = positions[:, 2]
        zs = positions[:, 1]

        # Update bones
        for i, (p, c) in enumerate(BODY_BONES):
            lines[i].set_data_3d(
                [xs[p], xs[c]],
                [ys[p], ys[c]],
                [zs[p], zs[c]],
            )

        # Update joint scatter (all joints)
        scatter._offsets3d = (xs, ys, zs)

        title_text.set_text(
            f"Frame #{receiver.frame_count}  |  {receiver.fps:.0f} FPS"
        )
        return lines + [scatter]

    _anim = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)  # ~30 Hz

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream BVH data from Noitom Axis Studio and visualise the skeleton."
    )
    parser.add_argument("--ip", default="127.0.0.1", help="Axis Studio IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7007, help="Axis Studio BVH port (default: 7007)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Axis Studio BVH Live Viewer")
    print(f"  Target: {args.ip}:{args.port}")
    print("=" * 60)
    print()

    receiver = BVHReceiver(ip=args.ip, port=args.port)
    receiver.start()

    try:
        run_live_plot(receiver)
    except KeyboardInterrupt:
        print("\nShutting down …")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
