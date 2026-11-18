"""--teacher-topk: the top-k + tail KL is exact at k = vocab, a lower bound below it, and close at large k."""

import torch

from calibration.train_map_legacy import TopKTeacher, kl_per_sequence, kl_topk_per_sequence


def _logprobs(*shape, scale=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).log_softmax(-1)


def test_topk_kl_bounds_full_kl():
    teacher, student = _logprobs(2, 7, 300), _logprobs(2, 7, 300, seed=1)
    full = kl_per_sequence(student, teacher)
    for k, exact in ((299, False), (100, False), (10, False)):
        store = TopKTeacher(k, "cpu")
        store.append_logprobs(teacher)
        values, idx, tail = store.batch([0], "cpu")
        approx = kl_topk_per_sequence(student, values, idx, tail)
        assert (approx <= full + 1e-3).all(), (k, approx, full)
    store = TopKTeacher(300, "cpu")
    store.append_logprobs(teacher)
    values, idx, tail = store.batch([0], "cpu")
    torch.testing.assert_close(kl_topk_per_sequence(student, values.float(), idx, tail),
                               kl_per_sequence(student, teacher), rtol=1e-2, atol=1e-3)
    exact_values, exact_idx = teacher.topk(300, dim=-1)
    torch.testing.assert_close(kl_topk_per_sequence(student, exact_values, exact_idx, tail), full,
                               rtol=1e-5, atol=1e-5)


def test_topk_kl_close_for_peaked_teacher():
    teacher, student = _logprobs(1, 16, 5000, scale=8.0), _logprobs(1, 16, 5000, scale=8.0, seed=2)
    store = TopKTeacher(1000, "cpu")
    store.append_logprobs(teacher)
    approx = kl_topk_per_sequence(student, *store.batch([0], "cpu"))
    full = kl_per_sequence(student, teacher)
    assert store.tail_mass[0] < 1e-3
    torch.testing.assert_close(approx, full, rtol=2e-2, atol=1e-3)


def test_topk_kl_gradient_matches_full_at_full_k():
    teacher = _logprobs(1, 5, 50)
    logits = torch.randn(1, 5, 50, requires_grad=True)
    values, idx = teacher.topk(50, dim=-1)
    tail = torch.full(values.shape[:-1], -1e30)
    g1, = torch.autograd.grad(kl_topk_per_sequence(logits.log_softmax(-1), values, idx, tail).sum(), logits)
    g2, = torch.autograd.grad(kl_per_sequence(logits.log_softmax(-1), teacher).sum(), logits)
    torch.testing.assert_close(g1, g2)
