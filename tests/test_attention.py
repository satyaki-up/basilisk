import torch
import torch.nn.functional as F

from basilisk.attention import ScaledDotProductAttention


def test_scaled_dot_product_attention_matches_torch():
    torch.manual_seed(0)
    dim = 4
    q_LD = torch.randn(3, dim)
    k_LD = torch.randn(3, dim)
    v_LD = torch.randn(3, dim)

    attention = ScaledDotProductAttention(dim)
    actual = attention(q_LD, k_LD, v_LD)
    expected = F.scaled_dot_product_attention(
        q_LD @ attention.w_q_DD,
        k_LD @ attention.w_k_DD,
        v_LD @ attention.w_v_DD,
    )

    torch.testing.assert_close(actual, expected)
