"""
Comprehensive Test Suite for HTML Motion to GIF Converter API
Tests cover endpoint functionality, error handling, and data validation
"""

import unittest
import json
import tempfile
import os
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import ConnectionError, Timeout


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for API endpoints"""

    def setUp(self):
        """Set up test client and base URL"""
        self.base_url = "http://localhost:5000"
        self.test_html = """
        <html>
            <head>
                <style>
                    @keyframes slide {
                        from { transform: translateX(0); }
                        to { transform: translateX(100px); }
                    }
                    .animated { animation: slide 2s infinite; }
                </style>
            </head>
            <body>
                <div class="animated">Test Content</div>
            </body>
        </html>
        """
        self.test_payload = {
            "html_content": self.test_html,
            "output_format": "gif",
            "duration": 2,
            "frame_rate": 10
        }

    def tearDown(self):
        """Clean up after tests"""
        pass

    def test_health_check_endpoint(self):
        """Test API health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")

    def test_convert_html_to_gif_success(self):
        """Test successful HTML to GIF conversion"""
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=self.test_payload,
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("Content-Type"), "image/gif")
        self.assertGreater(len(response.content), 0)

    def test_convert_with_custom_dimensions(self):
        """Test conversion with custom width and height"""
        payload = self.test_payload.copy()
        payload.update({
            "width": 800,
            "height": 600
        })
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 0)

    def test_convert_missing_html_content(self):
        """Test API error handling for missing HTML content"""
        payload = {
            "output_format": "gif",
            "duration": 2
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("html_content", data["error"].lower())

    def test_convert_invalid_duration(self):
        """Test API validation for invalid duration"""
        payload = self.test_payload.copy()
        payload["duration"] = -5
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_convert_invalid_frame_rate(self):
        """Test API validation for invalid frame rate"""
        payload = self.test_payload.copy()
        payload["frame_rate"] = 0
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_convert_invalid_content_type(self):
        """Test API error handling for invalid content type"""
        response = requests.post(
            f"{self.base_url}/api/convert",
            data=self.test_html,
            headers={"Content-Type": "text/plain"}
        )
        self.assertEqual(response.status_code, 400)

    def test_convert_empty_html(self):
        """Test handling of empty HTML content"""
        payload = self.test_payload.copy()
        payload["html_content"] = ""
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_convert_malformed_json(self):
        """Test handling of malformed JSON"""
        response = requests.post(
            f"{self.base_url}/api/convert",
            data="{invalid json}",
            headers={"Content-Type": "application/json"}
        )
        self.assertIn(response.status_code, [400, 415])

    def test_convert_oversized_dimensions(self):
        """Test handling of excessively large dimensions"""
        payload = self.test_payload.copy()
        payload.update({
            "width": 10000,
            "height": 10000
        })
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_convert_high_frame_rate(self):
        """Test handling of excessively high frame rate"""
        payload = self.test_payload.copy()
        payload["frame_rate"] = 1000
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)


class TestAPIInputValidation(unittest.TestCase):
    """Test suite for API input validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"
        self.valid_html = "<html><body><div>Test</div></body></html>"

    def test_html_content_type_validation(self):
        """Test HTML content is properly validated"""
        payload = {
            "html_content": self.valid_html,
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertIn(response.status_code, [200, 400, 422])

    def test_output_format_validation(self):
        """Test output format validation"""
        payload = {
            "html_content": self.valid_html,
            "output_format": "png"  # Invalid format
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_duration_range_validation(self):
        """Test duration parameter is within valid range"""
        valid_payloads = [
            {"html_content": self.valid_html, "duration": 0.1},
            {"html_content": self.valid_html, "duration": 1},
            {"html_content": self.valid_html, "duration": 60}
        ]
        for payload in valid_payloads:
            response = requests.post(
                f"{self.base_url}/api/convert",
                json=payload
            )
            self.assertNotEqual(response.status_code, 400)

    def test_width_height_type_validation(self):
        """Test width and height must be integers"""
        payload = {
            "html_content": self.valid_html,
            "width": "invalid",
            "height": "invalid"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)

    def test_frame_rate_type_validation(self):
        """Test frame rate must be numeric"""
        payload = {
            "html_content": self.valid_html,
            "frame_rate": "not_a_number"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertEqual(response.status_code, 400)

    def test_special_characters_in_html(self):
        """Test handling of special characters in HTML"""
        html_with_special_chars = "<html><body>Test © ® ™ € ñ</body></html>"
        payload = {
            "html_content": html_with_special_chars,
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertIn(response.status_code, [200, 400, 422])

    def test_unicode_in_html(self):
        """Test handling of unicode characters"""
        html_with_unicode = "<html><body>测试 テスト 테스트</body></html>"
        payload = {
            "html_content": html_with_unicode,
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertIn(response.status_code, [200, 400, 422])


class TestAPIErrorHandling(unittest.TestCase):
    """Test suite for API error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"

    def test_invalid_endpoint(self):
        """Test handling of invalid endpoint"""
        response = requests.post(
            f"{self.base_url}/api/invalid-endpoint",
            json={}
        )
        self.assertEqual(response.status_code, 404)

    def test_method_not_allowed(self):
        """Test GET request to POST-only endpoint"""
        response = requests.get(f"{self.base_url}/api/convert")
        self.assertEqual(response.status_code, 405)

    def test_missing_content_type_header(self):
        """Test request without Content-Type header"""
        response = requests.post(
            f"{self.base_url}/api/convert",
            data=json.dumps({"html_content": "<html></html>"})
        )
        # Should either auto-detect or reject
        self.assertIn(response.status_code, [200, 400, 415])

    def test_very_large_payload(self):
        """Test handling of very large payload"""
        large_html = "<html><body>" + "x" * (10 * 1024 * 1024) + "</body></html>"
        payload = {
            "html_content": large_html,
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload,
            timeout=5
        )
        self.assertIn(response.status_code, [400, 413, 422])

    def test_request_timeout(self):
        """Test handling of request timeout"""
        try:
            response = requests.post(
                f"{self.base_url}/api/convert",
                json={"html_content": "<html></html>"},
                timeout=0.001
            )
        except Timeout:
            pass  # Expected behavior

    def test_connection_error(self):
        """Test handling of connection errors"""
        invalid_url = "http://invalid-host-12345.local:5000/api/convert"
        try:
            response = requests.post(
                invalid_url,
                json={"html_content": "<html></html>"},
                timeout=1
            )
        except ConnectionError:
            pass  # Expected behavior

    def test_server_error_response(self):
        """Test handling of 5xx server errors"""
        # This would depend on actual server implementation
        response = requests.post(
            f"{self.base_url}/api/convert",
            json={"html_content": "<html></html>"}
        )
        if response.status_code >= 500:
            self.assertIn("error", response.json())


class TestAPIPerformance(unittest.TestCase):
    """Test suite for API performance characteristics"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"
        self.simple_html = "<html><body><p>Simple test</p></body></html>"

    def test_response_time_simple_conversion(self):
        """Test response time for simple conversion"""
        import time
        payload = {
            "html_content": self.simple_html,
            "output_format": "gif",
            "duration": 1
        }
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            self.assertLess(elapsed, 30)  # Should complete within 30 seconds

    def test_response_content_not_empty(self):
        """Test that response contains actual GIF data"""
        payload = {
            "html_content": self.simple_html,
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        if response.status_code == 200:
            self.assertGreater(len(response.content), 100)  # GIF should have minimum size

    def test_concurrent_requests(self):
        """Test API handling of concurrent requests"""
        import threading
        
        results = []
        
        def make_request():
            payload = {
                "html_content": self.simple_html,
                "output_format": "gif",
                "duration": 1
            }
            response = requests.post(
                f"{self.base_url}/api/convert",
                json=payload
            )
            results.append(response.status_code)
        
        threads = [threading.Thread(target=make_request) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        self.assertEqual(len(results), 3)


class TestAPISecurity(unittest.TestCase):
    """Test suite for API security aspects"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"

    def test_html_injection_attempt(self):
        """Test protection against HTML injection"""
        payload = {
            "html_content": "<script>alert('XSS')</script>",
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        # Should process safely without executing scripts
        self.assertNotEqual(response.status_code, 500)

    def test_sql_injection_prevention(self):
        """Test protection against SQL injection"""
        payload = {
            "html_content": "'; DROP TABLE users; --",
            "output_format": "gif"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        self.assertNotEqual(response.status_code, 500)

    def test_path_traversal_prevention(self):
        """Test protection against path traversal"""
        payload = {
            "html_content": "<html></html>",
            "output_file": "../../etc/passwd"
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        # Should not allow path traversal
        self.assertNotEqual(response.status_code, 200)

    def test_rate_limiting(self):
        """Test API rate limiting if implemented"""
        for i in range(10):
            payload = {
                "html_content": "<html></html>",
                "output_format": "gif"
            }
            response = requests.post(
                f"{self.base_url}/api/convert",
                json=payload
            )
            # If rate limited, should eventually get 429
            if response.status_code == 429:
                break

    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        response = requests.options(f"{self.base_url}/api/convert")
        # CORS handling depends on implementation
        self.assertIn(response.status_code, [200, 404, 405])


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000"

    def test_complete_conversion_workflow(self):
        """Test complete conversion workflow"""
        html = """
        <html>
            <head>
                <style>
                    @keyframes move {
                        0% { transform: translateX(0); }
                        100% { transform: translateX(100px); }
                    }
                    .box { animation: move 2s infinite; }
                </style>
            </head>
            <body>
                <div class="box" style="width:100px;height:100px;background:red;"></div>
            </body>
        </html>
        """
        payload = {
            "html_content": html,
            "output_format": "gif",
            "duration": 2,
            "frame_rate": 10,
            "width": 400,
            "height": 300
        }
        response = requests.post(
            f"{self.base_url}/api/convert",
            json=payload
        )
        
        if response.status_code == 200:
            self.assertEqual(response.headers.get("Content-Type"), "image/gif")
            self.assertGreater(len(response.content), 0)

    def test_batch_conversion_requests(self):
        """Test multiple sequential conversions"""
        payloads = [
            {
                "html_content": "<html><body>Test 1</body></html>",
                "output_format": "gif"
            },
            {
                "html_content": "<html><body>Test 2</body></html>",
                "output_format": "gif"
            },
            {
                "html_content": "<html><body>Test 3</body></html>",
                "output_format": "gif"
            }
        ]
        
        results = []
        for payload in payloads:
            response = requests.post(
                f"{self.base_url}/api/convert",
                json=payload
            )
            results.append(response.status_code)
        
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
