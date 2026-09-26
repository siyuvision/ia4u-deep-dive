import tempfile
import unittest
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from spatial_guard import fit_prior,apply_prior

class SpatialGuardTests(unittest.TestCase):
    def setUp(self):
        self.ct=np.full((240,120,120),-400,np.int16);self.ct[20:220,20:100,20:100]=0
        self.gt=np.zeros(self.ct.shape,np.uint8)
        self.gt[75:90,25:35,40:55]=1;self.gt[100:115,80:90,60:75]=1
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        a=sitk.GetImageFromArray(self.ct);a.SetSpacing((.21,)*3)
        b=sitk.GetImageFromArray(self.gt);b.CopyInformation(a)
        self.im=root/'ct.nii.gz';self.label=root/'ref.nii.gz';sitk.WriteImage(a,str(self.im));sitk.WriteImage(b,str(self.label))
        self.prior=fit_prior([('training',self.im,self.label)])
        self.direction=a.GetDirection()
    def tearDown(self):self.tmp.cleanup()
    def test_two_targets_preserved_remote_and_tiny_foreground_removed(self):
        raw=self.gt.copy();raw[180:190,40:55,40:55]=1;raw[90,60,60]=1
        actual,info=apply_prior(self.ct,raw,(.21,)*3,self.direction,self.prior)
        np.testing.assert_array_equal(actual,self.gt);self.assertEqual(info['kept_components'],2)
    def test_body_relative_translation(self):
        ct=np.pad(self.ct,((20,0),(0,0),(0,0)),constant_values=-400)
        pred=np.pad(self.gt,((20,0),(0,0),(0,0)))
        actual,_=apply_prior(ct,pred,(.21,)*3,self.direction,self.prior)
        np.testing.assert_array_equal(actual,pred)
    def test_empty_and_orientation_guard(self):
        actual,_=apply_prior(self.ct,np.zeros_like(self.gt),(.21,)*3,self.direction,self.prior)
        self.assertEqual(actual.sum(),0)
        with self.assertRaises(ValueError):apply_prior(self.ct,self.gt,(.21,)*3,(-1,0,0,0,-1,0,0,0,1),self.prior)

if __name__=='__main__':unittest.main()
