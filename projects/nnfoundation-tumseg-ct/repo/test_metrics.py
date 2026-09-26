import unittest
import numpy as np
from evaluate import overlap_metrics,surface_metrics

class MetricTests(unittest.TestCase):
    def test_empty_conventions(self):
        a=np.zeros((8,8,8),np.uint8);b=a.copy();b[2:4,2:4,2:4]=1
        self.assertEqual(overlap_metrics(a,a,(.21,)*3)['dice'],1)
        self.assertEqual(overlap_metrics(a,b,(.21,)*3)['dice'],0)
        self.assertIsNone(overlap_metrics(b,a,(.21,)*3)['recall'])
        self.assertIsNone(overlap_metrics(b,a,(.21,)*3)['signed_volume_error_percent'])
        self.assertIsNone(surface_metrics(a,b,(.21,)*3)['hd95_pooled_boundary_voxels_mm'])
    def test_units_and_two_lesions(self):
        a=np.zeros((10,10,10),np.uint8);a[1:3,1:3,1:3]=1;a[6:8,6:8,6:8]=1
        r=overlap_metrics(a,a,(.21,)*3)
        self.assertEqual(r['iou'],1);self.assertAlmostEqual(r['reference_volume_mm3'],16*.21**3)
    def test_boundary_spacing(self):
        a=np.zeros((8,8,8),np.uint8);b=a.copy();a[3,3,3]=1;b[4,3,3]=1
        self.assertAlmostEqual(surface_metrics(a,b,(.21,.21,.21))['hd95_pooled_boundary_voxels_mm'],.21)

if __name__=='__main__':unittest.main()
