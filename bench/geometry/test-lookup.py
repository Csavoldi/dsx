import importlib.util,pathlib,unittest
spec=importlib.util.spec_from_file_location('lookup',pathlib.Path(__file__).with_name('analyze-lookup.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class CausalLookup(unittest.TestCase):
    def setUp(self):self.prompt=list(range(1,9))+[90,91,92,99]+list(range(1,9))
    def test_full_and_rejected_blocks(self):
        self.assertEqual(m.audit(self.prompt,[90,91,92,99])['accepted_prefix_histogram'],[0,0,0,1])
        self.assertEqual(m.audit(self.prompt,[90,999])['accepted_prefix_histogram'],[1,0,0,0])
    def test_request_end_does_not_invent_acceptance(self):
        r=m.audit(self.prompt,[90,91]);self.assertEqual(r['attempts'],1);self.assertEqual(r['completed_blocks'],0);self.assertEqual(r['incomplete_blocks'],1)
    def test_unseen_future_and_outside_window_are_ineligible(self):
        self.assertEqual(m.audit(list(range(1,9)),[90,91,92,99])['attempts'],0)
        self.assertEqual(m.audit(self.prompt,[90,91,92,99],window=1)['attempts'],0)
if __name__=='__main__':unittest.main()
