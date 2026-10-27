import torch

from basilisk.optimizers.adamw import AdamW


def test_adamw_matches_torch():
    for weight_decay in (0.0, 1e-2):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        reference = torch.nn.Linear(4, 3)
        reference.load_state_dict(model.state_dict())

        optimizer = AdamW(model.parameters(), weight_decay=weight_decay)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), weight_decay=weight_decay)

        for step in range(10):
            torch.manual_seed(step)
            for param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
                grad = torch.randn_like(param)
                param.grad = grad
                reference_param.grad = grad.clone()
            optimizer.step()
            reference_optimizer.step()

        for param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, reference_param)
            state = optimizer.state[id(param)]
            reference_state = reference_optimizer.state[reference_param]
            torch.testing.assert_close(state["exp_avg"], reference_state["exp_avg"])
            torch.testing.assert_close(state["exp_avg_sq"], reference_state["exp_avg_sq"])
