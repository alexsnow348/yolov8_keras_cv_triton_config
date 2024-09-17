import json
from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def infer_yolov8_combined_ensemble(self):
        url = "/v2/models/yolov8_combined_ensemble/versions/1/infer"
        headers = {
            "Content-Type": "application/json"
        }
        with open("input_data_yolo.json", "r") as file:
            payload = json.load(file)
        self.client.post(url, headers=headers, json=payload)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)
    host = "http://localhost:8000"
