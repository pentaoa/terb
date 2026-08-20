import unittest
import numpy as np
import torch

from train import balance_by_dataset, loss_fn, shift_mel


class TrainingPipelineTests(unittest.TestCase):
    def test_shift_mel_does_not_wrap_edges(self):
        mel=np.zeros((2,5),np.float32)
        mel[:,0]=1
        up=shift_mel(mel,2)
        self.assertTrue(np.all(up[:,2]==1))
        self.assertTrue(np.all(up[:,:2]==0))
        mel[:]=0
        mel[:,-1]=1
        down=shift_mel(mel,-2)
        self.assertTrue(np.all(down[:,2]==1))
        self.assertTrue(np.all(down[:,3:]==0))

    def test_balance_by_dataset_equalizes_without_changing_ids(self):
        rows=[
            {"dataset":"large","recording_id":"a"},
            {"dataset":"large","recording_id":"b"},
            {"dataset":"small","recording_id":"c"},
        ]
        balanced=balance_by_dataset(rows)
        counts={name:sum(r["dataset"]==name for r in balanced) for name in ("large","small")}
        self.assertEqual(counts,{"large":2,"small":2})
        self.assertEqual({r["recording_id"] for r in balanced},{"a","b","c"})

    def test_missing_downbeat_mask_excludes_downbeat_loss(self):
        logits=torch.zeros(1,2,4)
        beat=torch.zeros(1,4)
        down=torch.ones(1,4)
        mask=torch.zeros(1,4)
        teacher=torch.zeros(1,4)
        two_head=loss_fn(logits,beat,down,mask,teacher,teacher,0.3)
        one_head=loss_fn(logits[:,:1],beat,down,mask,teacher,teacher,0.3)
        self.assertTrue(torch.allclose(two_head,one_head))


if __name__=="__main__":
    unittest.main()
