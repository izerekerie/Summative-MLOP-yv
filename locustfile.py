"""
Locust load testing file for Waste Classification FastAPI application.

This file defines load testing scenarios to simulate various user behaviors
and measure the performance of the FastAPI endpoints under different loads.

Usage:
    # Install locust first:
    pip install locust

    # Run with web UI (recommended):
    locust -f locustfile.py --host=http://0.0.0.0:8000

    # Run headless with specific parameters:
    locust -f locustfile.py --host=http://0.0.0.0:8000 --users 50 --spawn-rate 5 --run-time 60s --headless

    # Run with HTML report:
    locust -f locustfile.py --host=http://0.0.0.0:8000 --users 100 --spawn-rate 10 --run-time 120s --headless --html report.html
"""

import os
import random
import time
from io import BytesIO
from PIL import Image
import requests
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner


class WasteClassificationUser(HttpUser):
    """
    Simulates a user interacting with the Waste Classification API.
    
    This class defines various user behaviors and their relative weights
    to simulate realistic usage patterns.
    """
    
    # Wait time between requests (1-3 seconds to simulate human behavior)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts. Used for setup/login if needed."""
        # Check if backend is healthy before starting tests
        try:
            response = self.client.get("/health", timeout=10)
            if response.status_code != 200:
                print(f"Warning: Backend health check failed with status {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to backend: {e}")
    
    @task(10)  # Weight: 10 (most common operation)
    def health_check(self):
        """Test the health endpoint - most frequent operation."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(8)  # Weight: 8 (common operation)
    def get_classes(self):
        """Test getting the list of classes."""
        with self.client.get("/classes", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "classes" in data and len(data["classes"]) > 0:
                        response.success()
                    else:
                        response.failure("Classes endpoint returned empty or invalid data")
                except Exception as e:
                    response.failure(f"Failed to parse classes response: {e}")
            else:
                response.failure(f"Classes endpoint failed with status {response.status_code}")
    
    @task(6)  # Weight: 6 (prediction is resource-intensive)
    def predict_image(self):
        """Test image prediction endpoint with a synthetic image."""
        try:
            # Create a synthetic test image
            image = self._create_test_image()
            
            # Convert to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
            
            with self.client.post("/predict", files=files, catch_response=True, timeout=30) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "label" in data and "confidence" in data:
                            response.success()
                        else:
                            response.failure("Prediction response missing required fields")
                    except Exception as e:
                        response.failure(f"Failed to parse prediction response: {e}")
                else:
                    response.failure(f"Prediction failed with status {response.status_code}")
                    
        except Exception as e:
            self.client.get("/", catch_response=True).failure(f"Prediction test failed: {e}")
    
    @task(3)  # Weight: 3 (less frequent)
    def get_training_runs(self):
        """Test getting training runs."""
        with self.client.get("/retrain/runs?limit=10", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "runs" in data:
                        response.success()
                    else:
                        response.failure("Training runs response missing 'runs' field")
                except Exception as e:
                    response.failure(f"Failed to parse training runs response: {e}")
            else:
                response.failure(f"Training runs endpoint failed with status {response.status_code}")
    
    @task(2)  # Weight: 2 (less frequent)
    def get_example_image(self):
        """Test getting example images for classes."""
        classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]  # Common classes
        class_name = random.choice(classes)
        
        with self.client.get(f"/classes/{class_name}/example", catch_response=True) as response:
            if response.status_code == 200:
                # Check if we got image data
                if len(response.content) > 0:
                    response.success()
                else:
                    response.failure("Example image endpoint returned empty content")
            elif response.status_code == 404:
                # 404 is acceptable if no example exists
                response.success()
            else:
                response.failure(f"Example image endpoint failed with status {response.status_code}")

    
    def _create_test_image(self, size=(224, 224)):
        """Create a synthetic test image for testing purposes."""
        # Create a random colored image
        import random
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        image = Image.new('RGB', size, color)
        
        # Add some random noise to make it more realistic
        pixels = image.load()
        for i in range(0, size[0], 10):
            for j in range(0, size[1], 10):
                noise_color = (
                    min(255, max(0, color[0] + random.randint(-50, 50))),
                    min(255, max(0, color[1] + random.randint(-50, 50))),
                    min(255, max(0, color[2] + random.randint(-50, 50)))
                )
                pixels[i, j] = noise_color
        
        return image


class HighLoadUser(WasteClassificationUser):
    """
    High-intensity user that focuses on prediction endpoints.
    Use this for stress testing the model inference.
    """
    wait_time = between(0.5, 1.5)  # Faster requests
    
    @task(15)  # Much higher weight on predictions
    def predict_image(self):
        super().predict_image()
    
    @task(5)
    def health_check(self):
        super().health_check()
    
    @task(2)
    def get_classes(self):
        super().get_classes()


class LightUser(WasteClassificationUser):
    """
    Light user that mostly checks status and occasionally predicts.
    Simulates monitoring/dashboard users.
    """
    wait_time = between(3, 8)  # Slower, more realistic for monitoring
    
    @task(20)
    def health_check(self):
        super().health_check()
    
    @task(10)
    def get_training_runs(self):
        super().get_training_runs()
    
    @task(5)
    def get_classes(self):
        super().get_classes()
    
    @task(2)
    def predict_image(self):
        super().predict_image()


# Event handlers for custom reporting
@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Custom request handler to log specific metrics."""
    if exception:
        print(f"Request failed: {request_type} {name} - {exception}")
    elif response_time > 5000:  # Log slow requests (>5 seconds)
        print(f"Slow request detected: {request_type} {name} - {response_time}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("Starting Waste Classification API Load Test")
    print(f"Target host: {environment.host}")
    
    # Test connectivity
    try:
        response = requests.get(f"{environment.host}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Backend connectivity confirmed")
        else:
            print(f"⚠ Backend health check returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Could not connect to backend: {e}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("Load test completed")
    
    # Print summary statistics
    stats = environment.stats
    print(f"\nTest Summary:")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")
    
    if stats.total.num_failures > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure rate: {failure_rate:.2f}%")


# Custom test scenarios
class PredictionStressTest(HttpUser):
    """
    Focused stress test for prediction endpoint only.
    Use this to specifically test model inference performance.
    """
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        """Prepare test images."""
        self.test_images = []
        for i in range(5):  # Pre-generate 5 test images
            image = self._create_test_image()
            img_bytes = BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            self.test_images.append(img_bytes.getvalue())
    
    @task
    def predict_only(self):
        """Only test prediction endpoint."""
        try:
            img_data = random.choice(self.test_images)
            files = {'file': ('stress_test.jpg', BytesIO(img_data), 'image/jpeg')}
            
            with self.client.post("/predict", files=files, catch_response=True, timeout=30) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Prediction failed: {response.status_code}")
        except Exception as e:
            self.client.get("/", catch_response=True).failure(f"Stress test failed: {e}")
    
    def _create_test_image(self, size=(224, 224)):
        """Create a synthetic test image."""
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return Image.new('RGB', size, color)


if __name__ == "__main__":
    # This allows running the locust file directly for testing
    print("Locust file loaded successfully!")
    print("Available user classes:")
    print("- WasteClassificationUser: Balanced load testing")
    print("- HighLoadUser: High-intensity prediction testing")
    print("- LightUser: Light monitoring simulation")
    print("- PredictionStressTest: Prediction endpoint stress test")
    print("\nTo run tests:")
    print("locust -f locustfile.py --host=http://0.0.0.0:8000")
