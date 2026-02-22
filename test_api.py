import requests
import traceback

url = 'http://localhost:8000/api/remove-bg'
try:
    response = requests.post(url, files={'image': open('test_portrait.jpg', 'rb')})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        with open('result.png', 'wb') as f:
            f.write(response.content)
        print(f"Success! Output saved to result.png ({len(response.content)} bytes)")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print("Request failed")
    traceback.print_exc()
