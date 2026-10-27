from einops import rearrange
import torch
import torch.nn.functional as F

from basilisk.attention import MultiHeadAttention, ScaledDotProductAttention


def test_scaled_dot_product_attention_matches_torch():
    torch.manual_seed(0)
    dim = 4
    q_BLD = torch.randn(2, 3, dim)
    k_BLD = torch.randn(2, 3, dim)
    v_BLD = torch.randn(2, 3, dim)

    attention = ScaledDotProductAttention(dim)
    actual = attention(q_BLD, k_BLD, v_BLD)
    expected = F.scaled_dot_product_attention(
        q_BLD @ attention.w_q_DD,
        k_BLD @ attention.w_k_DD,
        v_BLD @ attention.w_v_DD,
    )

    torch.testing.assert_close(actual, expected)


def test_multi_head_attention_matches_torch():
    torch.manual_seed(0)
    dim = 4
    n_heads = 2
    x_BLD = torch.randn(2, 3, dim)

    attention = MultiHeadAttention(dim, n_heads, dim // n_heads)
    reference = torch.nn.MultiheadAttention(dim, n_heads, bias=False, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat(
                [
                    rearrange(attention.w_q_DHK, "d h k -> (h k) d"),
                    rearrange(attention.w_k_DHK, "d h k -> (h k) d"),
                    rearrange(attention.w_v_DHK, "d h k -> (h k) d"),
                ],
                dim=0,
            )
        )
        reference.out_proj.weight.copy_(rearrange(attention.w_o_HKD, "h k d -> d (h k)"))

    actual = attention(x_BLD)
    expected, _ = reference(x_BLD, x_BLD, x_BLD)

    torch.testing.assert_close(actual, expected)
