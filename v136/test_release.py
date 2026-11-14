"""Integration checks for the standalone English distribution."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from .service import Handler

class ReleaseHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def request(self, path, payload=None):
        request = urllib.request.Request(self.base + path, data=payload,
                                         headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=30)

    def test_english_page_and_links(self):
        with self.request('/') as response:
            html = response.read().decode()
        self.assertIn('lang="en"', html)
        self.assertNotIn('href="/report135"', html)
        self.assertNotIn('href="/bench133"', html)
        for path in ['/report136', '/artifacts136/hardware_bench/bench_results.csv',
                     '/docs_v136/ARCHITECTURE.md']:
            with self.request(path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())

    def test_get_post_parity(self):
        q = {'person': 'C135-108', 'task': '13-16', 'stage': 'after', 'lang': 'en'}
        with self.request('/api/v135?person=C135-108&task=13-16&stage=after&lang=en') as r:
            expected = json.load(r)
        with self.request('/api/v135', json.dumps(q).encode()) as r:
            self.assertEqual(json.load(r), expected)

    def test_default_language_english(self):
        with self.request('/api/v135?person=C135-108&task=13-16') as r:
            self.assertEqual(json.load(r)['lang'], 'en')

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for raw in [b'{"task":"13-16","task":"13-16"}', b'{"task":NaN}',
                    b'{"task":Infinity}', b'[]']:
            with self.subTest(raw=raw), self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/v135', raw)
            self.assertEqual(error.exception.code, 400)

    def test_unknown_route_and_traversal(self):
        for path, status in [('/unknown', 404),
                             ('/artifacts136/../models_v135/study/twin.joblib', 400),
                             ('/docs_v136/../README.md', 400)]:
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, status)

if __name__ == '__main__':
    unittest.main(verbosity=2)
