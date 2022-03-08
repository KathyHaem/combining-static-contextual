"""
Adapted from https://github.com/arminarj/DeepGCCA-pytorch/ by Katharina Hämmerl.

MIT License

Copyright (c) 2020 Armin Arjmand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch


def dgcca_loss(H_list):
    r = 1e-4
    eps = 1e-8
    top_k = 10
    AT_list = []

    for H in H_list:
        assert torch.isnan(H).sum().item() == 0

        o_shape = H.size(0)  # N
        m = H.size(1)  # out_dim

        Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(Hbar).sum().item() == 0

        A, S, B = Hbar.svd(some=True, compute_uv=True)
        A = A[:, :top_k]
        assert torch.isnan(A).sum().item() == 0

        S_thin = S[:top_k]
        S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)
        assert torch.isnan(S2_inv).sum().item() == 0

        T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
        assert torch.isnan(T2).sum().item() == 0
        T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device))

        T = torch.diag(torch.sqrt(T2))
        assert torch.isnan(T).sum().item() == 0

        T_unnorm = torch.diag(S_thin + eps)
        assert torch.isnan(T_unnorm).sum().item() == 0

        AT = torch.mm(A, T)
        AT_list.append(AT)

    M_tilde = torch.cat(AT_list, dim=1)
    assert torch.isnan(M_tilde).sum().item() == 0

    _, S, _ = M_tilde.svd(some=True)
    assert torch.isnan(S).sum().item() == 0

    use_all_singular_values = False
    if not use_all_singular_values:
        S = S[:top_k]

    corr = torch.sum(S)
    assert torch.isnan(corr).item() == 0

    loss = - corr
    return loss
