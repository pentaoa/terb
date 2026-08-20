import unittest
import torch
from causal_beat_this import LOOKAHEAD_FRAMES,load_causal_checkpoint


class CausalBeatThisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_causal_checkpoint(
            "data/beat_this-reference",
            "data/beat_this-models/small0.ckpt",
            context_frames=28,
        ).eval()

    def test_future_changes_do_not_cross_four_frame_lookahead(self):
        torch.manual_seed(7)
        frames=96
        split=64
        a=torch.randn(1,frames,128)
        b=a.clone()
        b[:,split:]=torch.randn_like(b[:,split:])
        with torch.inference_mode():
            pa=self.model(a)["beat"]
            pb=self.model(b)["beat"]
        safe_end=split-LOOKAHEAD_FRAMES
        err=(pa[:,:safe_end]-pb[:,:safe_end]).abs().max().item()
        changed=(pa[:,split:]-pb[:,split:]).abs().max().item()
        self.assertLess(err,2e-5)
        self.assertGreater(changed,1e-3)


    def test_input_older_than_five_seconds_cannot_affect_current_output(self):
        torch.manual_seed(11)
        frames=360
        a=torch.randn(1,frames,128)
        b=a.clone()
        b[:,:80]=torch.randn_like(b[:,:80])
        with torch.inference_mode():
            pa=self.model(a)["beat"]
            pb=self.model(b)["beat"]
        err=(pa[:,-1]-pb[:,-1]).abs().max().item()
        self.assertLess(err,2e-5)


if __name__=="__main__":
    unittest.main()
