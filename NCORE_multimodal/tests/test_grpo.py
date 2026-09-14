import torch
from ncore.rl.grpo import group_advantages, clipped_grpo_loss

def test_group_advantage_zero_mean():
    r=torch.tensor([1.,2.,3.,4., 4.,3.,2.,1.])
    a=group_advantages(r,4).view(2,4)
    assert torch.allclose(a.mean(1),torch.zeros(2),atol=1e-5)

def test_grpo_loss_finite():
    x=torch.randn(8,requires_grad=True); old=x.detach().clone(); adv=torch.randn(8); ent=torch.rand(8)
    loss=clipped_grpo_loss(x,old,adv,ent); assert torch.isfinite(loss); loss.backward(); assert x.grad is not None
