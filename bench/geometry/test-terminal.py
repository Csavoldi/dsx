import unittest
from terminal import Screen,certify,split_frames
class TerminalCertification(unittest.TestCase):
    def test_duplicate_home_does_not_create_a_frame(self):
        payload='\r\n'.join(['⠿'*10+'⣿'*10]*8)
        one='\x1b[H'+payload
        doubled='\x1b[H'+payload+'\x1b[H'
        frames=split_frames(doubled*8)
        self.assertEqual(frames,split_frames(one*8))
        self.assertTrue(certify(frames,20,10)['pass'])
    def test_space_only_frame_is_preserved(self):
        frames=split_frames('\x1b[H⣿\x1b[H \x1b[H')
        self.assertEqual(frames,['⣿',' '])
        self.assertEqual(split_frames('\x1b[H\x1b[0m\r\n\x1b[H'),[])
    def test_duplicate_home_does_not_hide_real_trails(self):
        long='\r\n'.join(['⠿'*10+'⣿'*10]*8)
        short='\r\n'.join(['⠿'*5+'⣿'*5]*8)
        frames=split_frames(''.join('\x1b[H'+x+'\x1b[H' for x in [long]*3+[short]*3))
        self.assertGreater(certify(frames,24,10)['max_ghost_fraction'],.1)
    def test_clearing_removes_trails(self):
        s=Screen(10,4);s.feed('\x1b[H'+'⣿'*6);s.feed('\x1b[H'+'⣿'*2);self.assertEqual(len(s.dots()),6)
        s.feed('\x1b[H'+'⣿'*2+'\x1b[K');self.assertEqual(len(s.dots()),2)
        s.feed('\x1b[2J');self.assertFalse(s.dots())
    def test_rgb_and_delayed_wrap(self):
        s=Screen(4,3);s.feed('abcd\r\nefgh');self.assertEqual(s.cells[1][0][0],'e')
        s.feed('\x1b[H\x1b[38;2;100;150;200m⣿');self.assertEqual(s.cells[0][0][1],(100,150,200))
    def test_shaded_frames_pass_and_uniform_faces_do_not(self):
        shaded='\x1b[2J'+('\r\n'.join(['⠿'*10+'⣿'*10]*8))
        self.assertTrue(certify([shaded]*8,20,10)['pass'])
        uniform='\x1b[2J'+('\r\n'.join(['⣿'*20]*8))
        self.assertFalse(certify([uniform]*8,20,10)['pass'])
    def test_trails_are_not_certified(self):
        long='\r\n'.join(['⠿'*10+'⣿'*10]*8)
        short='\r\n'.join(['⠿'*5+'⣿'*5]*8)
        self.assertGreater(certify([long,long,long,short,short,short],24,10)['max_ghost_fraction'],.1)
if __name__=='__main__':unittest.main()
