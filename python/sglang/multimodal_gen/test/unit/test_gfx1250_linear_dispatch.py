# SPDX-License-Identifier: Apache-2.0
"""On gfx1250, bf16 linears whose output passes 4 GiB must stay correct.

tgemm corrupts output rows past 2**31 bytes and FlyDSL's a16w16 rows past 2**32
bytes, both without faulting; at Wan2.2 720p that turned the last video frames
into noise."""

import unittest

import pytest
import torch

from sglang.multimodal_gen.runtime.layers import linear
from sglang.srt.utils import is_gfx1250_supported

# Wan2.2 A14B 720p 193-frame FFN up-projection: the output is 4.88 GB.
M, N, K = 176400, 13824, 5120


class TestGfx1250LargeOutputLinear(unittest.TestCase):
    def test_rows_past_4gib_output_match_fp32_reference(self):
        aiter_gemm = linear.USE_AITER_GFX1250 or linear._USE_AITER_TGEMM
        if not (aiter_gemm and is_gfx1250_supported()):
            pytest.skip("needs gfx1250 with SGLANG_USE_AITER=1 or AITER tgemm")
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = linear.apply_unquantized_linear(x, w, b)
        # Rows 77672 and 155344 are the first past 2**31 and 2**32 output bytes.
        rows = torch.cat(
            [torch.arange(0, 64), torch.arange(78000, 78064), torch.arange(M - 64, M)]
        ).cuda()
        ref = x[rows].float() @ w.float().t() + b.float()
        torch.testing.assert_close(out[rows].float(), ref, atol=0.05, rtol=0.02)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
