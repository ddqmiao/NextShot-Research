"""Synthetic CPU checks; no pretrained weights, personal data, or network access."""
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from nextshot.backbone import SourceEncoder
from nextshot.decoder import CoarseDecoder
from nextshot.framing import decode_geometry
from nextshot.losses import loss
from nextshot.model import reason_losses


def configuration():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/nextshot_4b.json').read_text())


def graph():
    g = dict(visual=torch.randn(1, 3, 2048), geometry=torch.zeros(1, 3, 14),
             orientation=torch.zeros(1, 3, 12), body=torch.randn(1, 3, 533),
             node_mask=torch.tensor([[True, True, False]]),
             semantic_match=torch.eye(3)[None], person=torch.tensor([[True, False, False]]),
             focus_hint=torch.zeros(1, 3))
    g['geometry'][..., :2] = .5
    g['geometry'][..., 2] = torch.tensor(.08).log()
    g['geometry'][..., 3:5] = .5
    g['geometry'][..., 5:7] = .4
    return g


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_zero_residual_geometry(self):
        g = graph()['geometry']
        boxes, centers, areas = decode_geometry(torch.zeros(1, 3, 7), g)
        torch.testing.assert_close(boxes, torch.tensor([.3, .3, .7, .7]).expand_as(boxes))
        torch.testing.assert_close(centers, torch.full_like(centers, .5))
        torch.testing.assert_close(areas, torch.full_like(areas, .08))

    def test_geometry_bounds(self):
        g = graph()['geometry']
        boxes, centers, areas = decode_geometry(torch.randn(1, 3, 7)*8, g)
        self.assertTrue(((boxes >= 0) & (boxes <= 1)).all())
        self.assertTrue((boxes[..., 2:] >= boxes[..., :2]).all())
        self.assertTrue(((centers >= boxes[..., :2]-1e-6) & (centers <= boxes[..., 2:]+1e-6)).all())
        self.assertTrue((areas <= (boxes[..., 2:]-boxes[..., :2]).prod(-1)+1e-6).all())

    def test_masking_and_conditioning_gradients(self):
        c, g = configuration(), graph()
        encoder, decoder = SourceEncoder(c), CoarseDecoder(c, 16)
        source, frame, relations = encoder(g)
        output = decoder(torch.randn(1, 3, 16), torch.randn(1, 16), source, g, relations)
        self.assertEqual(output['boxes'].shape, (1, 3, 4))
        self.assertEqual(output['emphasis'][0, 2].item(), 0.)
        value = output['boxes'][:, :2].sum() + output['areas'][:, :2].sum()
        value.backward()
        for layer in (decoder.frame_condition, decoder.coarse_condition, decoder.frame_head, decoder.coarse_head):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_complete_reference_selection(self):
        # Each independently optimal region comes from a different reference.
        # The loss must retain one complete reference and its associated body label.
        boxes = torch.tensor([[[.1, .1, .3, .3], [.6, .6, .9, .9]]], requires_grad=True)
        output = dict(boxes=boxes, centroids=(boxes[..., :2]+boxes[..., 2:])/2,
                      areas=torch.tensor([[.03, .06]], requires_grad=True),
                      log_aspect=torch.zeros(1, requires_grad=True),
                      framing_logits=torch.zeros(1, 2, 6),
                      coarse_framing_logits=torch.tensor([[[2., 0.], [0., 0.]]], requires_grad=True))
        a = dict(reference_id='reference_a', valid=[True, True],
                 boxes=[[.1, .1, .3, .3], [.2, .2, .4, .4]], centroids=[[.2, .2], [.3, .3]],
                 areas=[.03, .03], log_aspect=0., body=[None, None], coarse_body=[0, None])
        b = dict(reference_id='reference_b', valid=[True, True],
                 boxes=[[.6, .6, .9, .9], [.6, .6, .9, .9]], centroids=[[.75, .75], [.75, .75]],
                 areas=[.06, .06], log_aspect=0., body=[None, None], coarse_body=[1, None])
        target = dict(region_ids=['r1', 'r2'], references=[a, b])
        value, details = loss(output, [target], {'person':torch.tensor([[True, False]])})
        selected = next(r for r in (a, b) if r['reference_id'] == details[0]['reference_id'])
        single, _ = loss(output, [dict(target, references=[selected])], {'person':torch.tensor([[True, False]])})
        torch.testing.assert_close(value, single, rtol=0, atol=0)
        self.assertGreater(details[0]['geometry'], 0)
        self.assertEqual(details[0]['coarse_supervised_labels'], [selected['coarse_body'][0]])
        value.backward()
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_chunked_reason_loss_and_empty_answer(self):
        h = torch.randn(2, 6, 8, requires_grad=True)
        w = torch.randn(13, 8)
        labels = torch.tensor([[-100, -100, 3, 4, 5, 6], [-100]*6])
        values, counts = reason_losses(h, labels, w, 2)
        expected = F.cross_entropy(F.linear(h[0, 1:5], w), labels[0, 2:6])
        torch.testing.assert_close(values[0], expected)
        self.assertEqual(counts, [4, 0])
        self.assertEqual(values[1].item(), 0)
        values.sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())


if __name__ == '__main__':
    unittest.main()
