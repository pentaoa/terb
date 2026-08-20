import torch
from train_streaming_crnn import StreamingBeatCRNN


def test_streaming_matches_full_sequence():
    torch.manual_seed(3); model = StreamingBeatCRNN().eval(); x = torch.randn(2, 41, 128)
    with torch.no_grad():
        full, _ = model(x); state = None; pieces = []
        for part in x.split([5, 7, 3, 11, 15], dim=1):
            y, state = model(part, state); pieces.append(y)
    assert torch.max(torch.abs(full - torch.cat(pieces, dim=2))).item() < 2e-5


def test_future_does_not_change_past():
    torch.manual_seed(4); model = StreamingBeatCRNN().eval(); x = torch.randn(1, 32, 128); changed = x.clone(); changed[:, 20:] += 10
    with torch.no_grad(): a, _ = model(x); b, _ = model(changed)
    assert torch.equal(a[:, :, :20], b[:, :, :20])
