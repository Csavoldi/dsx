import contextlib,json,os,pathlib,socket,subprocess,sys,tempfile,time,unittest
from unittest.mock import patch
import run
HERE=pathlib.Path(__file__).resolve().parent
class ProxyOwnership(unittest.TestCase):
    def test_existing_listener_is_not_accepted_as_the_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            def start(p,nonce):
                return subprocess.Popen([sys.executable,str(HERE/'proxy.py')],env=dict(os.environ,GAUNTLET_RUN=str(p),GAUNTLET_PORT=str(port),GAUNTLET_SEED_BASE='1',GAUNTLET_PROXY_NONCE=nonce),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            first=start(a,'first');second=None
            try:
                run.wait_proxy(first,a,port,'first');second=start(b,'second')
                with self.assertRaisesRegex(RuntimeError,'Owned proxy'):run.wait_proxy(second,b,port,'second')
                self.assertIsNone(first.poll(),'A foreign listener must not be killed')
            finally:
                for p in [second,first]:
                    if p is not None:
                        if p.poll() is None:p.terminate()
                        p.wait(timeout=5)
    def test_later_foreign_requests_are_excluded_but_owned_late_end_is_retained(self):
        records=[{'number':1,'start':9,'event':'start'}, {'number':2,'start':10,'event':'start'}, {'number':2,'start':10,'event':'end','end':31}, {'number':3,'start':21,'event':'end'}]
        owned,outside=run.metric_window(records,{'wall':10,'agent_wall_s':10})
        self.assertEqual([x['number'] for x in owned],[2,2])
        self.assertEqual([x['number'] for x in outside],[1,3])
    def test_missing_pane_cannot_leak_the_owned_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);proxy=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);a=(root/'a.log').open('w');b=(root/'b.log').open('w')
            try:
                failure=subprocess.CompletedProcess([],1,'','missing pane')
                with patch.object(run.subprocess,'run',return_value=failure):run.cleanup_run(None,proxy,'%fixture','fixture',a,b,root)
                self.assertIsNotNone(proxy.poll());self.assertEqual(len(json.loads((root/'cleanup.json').read_text())['errors']),2)
            finally:
                if proxy.poll() is None:proxy.kill();proxy.wait(timeout=5)
if __name__=='__main__':unittest.main()
