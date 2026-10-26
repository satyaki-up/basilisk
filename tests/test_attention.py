from einops import rearrange
import torch
import torch.nn.functional as F

from basilisk.attention import MultiHeadAttention, ScaledDotProductAttention


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


def test_multi_head_attention_matches_torch():
    torch.manual_seed(0)
    dim = 4
    n_heads = 2
    q_LD = torch.randn(3, dim)
    k_LD = torch.randn(3, dim)
    v_LD = torch.randn(3, dim)

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

    actual = attention(q_LD, k_LD, v_LD)
    expected, _ = reference(q_LD.unsqueeze(0), k_LD.unsqueeze(0), v_LD.unsqueeze(0))

    torch.testing.assert_close(actual, expected.squeeze(0))
