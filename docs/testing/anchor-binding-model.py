# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Enumerate RFC-0074's public byte partition, token rows and block bound.

Run: python3 -I docs/testing/anchor-binding-model.py
This assumes verified logs in canonical frame order. It does not model log
validation, AES/SHA, proof acceptance, masking or a proof-size upper bound.
"""

from collections import Counter
from itertools import product


def mapping(requests):
    """Requests are (session_id, key_commitment, token_counts_per_frame)."""
    stride = 1 << (max(sum(r[2]) for r in requests) - 1).bit_length()
    sessions = {}
    for request, (session_id, key, frames) in enumerate(requests):
        session = sessions.setdefault((session_id, key), [[], [[], [], [], []], [], 0])
        public, limbs, rows, offset = session
        position = 0
        for tokens in frames:
            length = 22 + 4 * tokens
            padded = (length + 15) // 16 * 16
            public.extend(range(offset, offset + 22))
            public.extend(range(offset + length, offset + padded))
            for token in range(tokens):
                rows.append(request * stride + position)
                position += 1
                for byte in range(4):
                    limbs[byte].append(offset + 22 + 4 * token + byte)
            offset += padded
        session[3] = offset
    return stride, sessions


def check(requests, result):
    """Judge coverage by counting positions, separately from the mapping loops."""
    stride, sessions = result
    expected_stride = 1
    while expected_stride < max(sum(r[2]) for r in requests):
        expected_stride *= 2
    assert stride == expected_stride, "canonical request slot stride"
    covered = []
    expected_sessions = list(dict.fromkeys((r[0], r[1]) for r in requests))
    assert list(sessions) == expected_sessions
    plaintexts = {session: bytearray() for session in expected_sessions}
    token_values = {}
    for request, (session_id, key, frames) in enumerate(requests):
        position = 0
        for count in frames:
            frame = bytearray(22)
            for _ in range(count):
                row = request * stride + position
                value = ((row + 1) * 0x01020305) % (1 << 32)
                token_values[row] = value
                frame.extend(value.to_bytes(4, "little"))
                position += 1
            frame.extend(bytes(-len(frame) % 16))
            plaintexts[(session_id, key)].extend(frame)
    for session, (public, limbs, rows, size) in sessions.items():
        counts = Counter(public)
        for offsets in limbs:
            assert len(offsets) == len(rows)
            counts.update(offsets)
        assert counts == Counter(range(size)), "plaintext partition"
        covered.extend(rows)
        plaintext = plaintexts[session]
        assert len(plaintext) == size
        for token, row in enumerate(rows):
            value = sum(plaintext[limbs[b][token]] << (8 * b) for b in range(4))
            assert value == token_values.get(row), "token byte order"
    expected_rows = Counter(
        request * stride + token
        for request, (_, _, frames) in enumerate(requests)
        for token in range(sum(frames))
    )
    assert Counter(covered) == expected_rows, "token row coverage"


def main():
    cases = 0
    # All small frame sequences with a token and at most two empty frames.
    for length in range(1, 7):
        for frames in product(range(4), repeat=length):
            tokens = sum(frames)
            if not tokens or tokens > 6 or frames.count(0) > 2:
                continue
            blocks = sum((72 + 4 * n + 63) // 64 for n in frames)
            assert blocks <= 2 * tokens + 4, "SHA-256 frame bound"
            assert sum((22 + 4 * n + 15) // 16 for n in frames) <= 4 * blocks
            # A shared session, a distinct session, and a reopened id/new key.
            for other in [(7, 11), (8, 11), (7, 12)]:
                requests = [(7, 11, frames), (*other, (1,)), (7, 11, (2,))]
                check(requests, mapping(requests))
                cases += 1

    # The per-frame inequality is affine between changes in the block ceiling.
    for tokens in range(1, 65537):
        assert (72 + 4 * tokens + 63) // 64 <= 2 * tokens
    assert 2 * 2045 + 5 <= 4096 < 2 * 2046 + 5
    assert (2 * 1021 + 5) + (2 * 1022 + 5) == 4096
    assert sum((72 + 4 * n + 63) // 64 for n in (100, 0, 0, 0)) == 14
    check([(7, 11, (100, 0, 0, 0))], mapping([(7, 11, (100, 0, 0, 0))]))
    assert (2 * 1 + 5) * 585 <= 4096 < (2 * 1 + 5) * 586

    # Negative controls: each omission/duplication must violate its own invariant.
    requests = [(7, 11, (0, 1, 2)), (7, 11, (1,))]
    for fault in range(6):
        result = mapping(requests)
        public, limbs, rows, _ = next(iter(result[1].values()))
        if fault == 0:
            public.pop()
        elif fault == 1:
            limbs[0].append(limbs[0][0])
        elif fault == 2:
            rows[0] = result[0] - 1  # A padded token row.
        elif fault == 3:
            rows[1] = rows[0]
        elif fault == 4:
            limbs[0], limbs[1] = limbs[1], limbs[0]
        else:
            stride = result[0]
            rows[:] = [(row // stride) * (2 * stride) + row % stride for row in rows]
            result = (2 * stride, result[1])
        try:
            check(requests, result)
        except AssertionError:
            continue
        raise AssertionError(f"negative control {fault} survived")
    print(f"PASS: {cases} byte/row partitions, 65536 frame bounds, 6 negative controls")


if __name__ == "__main__":
    main()
