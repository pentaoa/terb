import torch
from causal_beat_this import load_causal_checkpoint
from train_frequency_gru import FrequencyGRU


def main():
    teacher = load_causal_checkpoint("data/beat_this-reference", "runs/local84-small0-guitar-pilot/epoch-002.ckpt", "cpu", 84)
    torch.manual_seed(7); model = FrequencyGRU(teacher).eval(); x = torch.randn(1, 23, 128)
    with torch.no_grad():
        full, _ = model(x); state = None; parts = []
        for item in x.split([5, 3, 7, 8], 1): y, state = model(item, state); parts.append(y)
        changed = x.clone(); changed[:, 15:] += 5; a, _ = model(x); b, _ = model(changed)
    assert torch.max(torch.abs(full - torch.cat(parts, 2))).item() < 3e-5
    assert torch.equal(a[:, :, :15], b[:, :, :15])
    print("frequency-GRU streaming equivalence and causality passed")


if __name__ == "__main__": main()
