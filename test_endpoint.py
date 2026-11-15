import os
import requests
import json

def test_dashboard_endpoint():
    port = int(os.environ.get("PORT", 5050))
    url = f"http://127.0.0.1:{port}/super-resolve"
    lr1_path = "sample_lr1.png"
    lr2_path = "sample_lr2.png"

    if not os.path.exists(lr1_path) or not os.path.exists(lr2_path):
        print(f"Error: {lr1_path} or {lr2_path} not found in current directory.")
        return False

    print(f"Sending POST request to {url}...")
    try:
        with open(lr1_path, "rb") as f1, open(lr2_path, "rb") as f2:
            files = {
                "lr1": (os.path.basename(lr1_path), f1, "image/png"),
                "lr2": (os.path.basename(lr2_path), f2, "image/png")
            }
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            print("Success! Response 200 OK received.")
            data = response.json()
            
            # Check keys
            expected_keys = ["hr_image", "distortion_report", "metrics", "decision_report"]
            missing_keys = [k for k in expected_keys if k not in data]
            if missing_keys:
                print(f"Warning: Response is missing expected keys: {missing_keys}")
                return False
            
            print("\nResponse structure verified:")
            print(f"  - hr_image (base64, length: {len(data['hr_image'])})")
            print(f"  - metrics: {data['metrics']}")
            print(f"  - distortion_report length: {len(data['distortion_report'])} characters")
            print(f"  - decision_report length: {len(data['decision_report'])} characters")
            return True
        else:
            print(f"Failed! Server returned status code {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Flask server. Is app.py running on http://127.0.0.1:5000?")
        return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    test_dashboard_endpoint()
