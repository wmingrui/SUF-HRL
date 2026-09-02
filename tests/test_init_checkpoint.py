import torch

from tools.train import load_initial_weights


def test_load_initial_weights_restores_clean_state_dict(tmp_path):
    src = torch.nn.Linear(3, 2)

    ckpt = tmp_path / "clean.pth"
    torch.save(src.state_dict(), ckpt)

    dst = torch.nn.Linear(3, 2)

    with torch.no_grad():
        for p in dst.parameters():
            p.zero_()

    load_initial_weights(dst, ckpt)

    src_state = src.state_dict()
    dst_state = dst.state_dict()

    assert src_state.keys() == dst_state.keys()

    for k in src_state:
        assert torch.equal(src_state[k], dst_state[k])
