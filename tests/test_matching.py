import unittest
import numpy as np
from PIL import Image
from data_processing.semantics import evidence
from data_processing.matching import match_regions,partial_assignment
from data_processing.regions import crop_box,letterbox,FeatureConfig


def r(rid,text,group='person',primary=False):
    return dict(region_id=rid,semantic=text,semantic_group=group,primary=primary)


class V2Tests(unittest.TestCase):
    def match(self,a,b,v,c=None):
        pair=dict(pair_id='p',worse_image_id='a',better_image_id='b')
        return match_regions(pair,a,b,v,v if c is None else c,'v2')[0]

    def test_pose_change_can_match_without_geometry(self):
        rows=self.match([r('R1','woman lying on grass',primary=True)],[r('R5','woman standing on grass',primary=True)],[[.85]])
        self.assertEqual(rows[0]['match_status'],'matched')
        self.assertEqual(rows[0]['target_region_id'],'R5')

    def test_primary_prior_is_not_proof(self):
        rows=self.match([r('R1','woman',primary=True)],[r('R2','woman',primary=True)],[[.25]])
        self.assertEqual(rows[0]['match_status'],'ambiguous')

    def test_crowd_to_individual_never_auto_confirmed(self):
        rows=self.match([r('R4','group of people in foreground',primary=True)],[r('R1','woman in black dress',primary=True)],[[.99]])
        self.assertEqual(rows[0]['match_status'],'ambiguous')

    def test_flowers_not_trees_despite_same_group_and_visual(self):
        a=[r('flowers','pink flowers and green grass','vegetation'),r('woods','dense green trees and hillside','vegetation')]
        b=[r('field','field of pink flowers and green grass','vegetation')]
        rows=self.match(a,b,[[.9],[.95]])
        self.assertIn(('flowers','field','matched'),[(x['source_region_id'],x['target_region_id'],x['match_status']) for x in rows])
        self.assertIn(('woods',None,'source_only'),[(x['source_region_id'],x['target_region_id'],x['match_status']) for x in rows])

    def test_soft_category_boundary_and_pose_words(self):
        self.assertTrue(evidence(r('a','glass wall and reflection','interior'),r('b','glass wall','architecture'))['compatible'])
        self.assertTrue(evidence(r('a','woman lying'),r('b','woman standing'))['compatible'])
        self.assertFalse(evidence(r('a','woman'),r('b','clear sky','sky'))['compatible'])

    def test_dummy_assignment_does_not_force_bad_edge(self):
        edges,_=partial_assignment(np.array([[.9,.7],[.65,.1]]),np.ones((2,2),bool),.6)
        self.assertEqual(edges,{(0,0)})
        rows=self.match([r('a','woman'),r('b','woman')],[r('c','woman')],[[.9],[.9]])
        self.assertEqual({x['match_status'] for x in rows},{'ambiguous'})

    def test_crop_preserves_aspect_and_excludes_padding(self):
        rgb=Image.new('RGB',(100,300));mask=np.zeros((300,100),bool);mask[30:230,20:70]=1
        box=crop_box(mask,0)
        self.assertEqual(box,(20,30,70,230))
        image,fg,valid,transform=letterbox(rgb,mask,box,FeatureConfig())
        self.assertEqual(transform['resized_wh'],[112,448])
        self.assertEqual(image.shape,(448,112,3))
        self.assertTrue(np.all(fg[valid==0]==0))
        self.assertAlmostEqual(transform['scale_xy'][0],transform['scale_xy'][1])

    def test_empty_mask_and_nonfinite_evidence_rejected(self):
        with self.assertRaises(ValueError):crop_box(np.zeros((20,30)),.1)
        with self.assertRaises(ValueError):self.match([r('a','woman')],[r('b','woman')],[[float('nan')]])


if __name__=='__main__':unittest.main()
