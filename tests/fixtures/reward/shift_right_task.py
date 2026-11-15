import torch


def shift_right(x):
    """x[1:] = x[:-1], in place; returns x."""
    x[1:] = x[:-1].clone()
    return x


##################################################################################################################################################


def test_shift_right():
    results = {}
    for i, n in enumerate((64, 100, 7)):
        x = torch.arange(n, dtype=torch.float32, device="cuda")
        results[f"test_case_{i + 1}"] = shift_right(x).clone()
    return results


result_gold = test_shift_right()
