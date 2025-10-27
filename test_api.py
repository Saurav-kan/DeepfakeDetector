import requests

# The URL of the local API endpoint
url = "http://127.0.0.1:8000/predict/"

# The path to the image you want to test
# Make sure you have an image file (e.g., "test_image.jpg") in the same directory
image_path = "path/to/your/image.jpg" # <-- IMPORTANT: CHANGE THIS PATH

try:
    with open(image_path, "rb") as image_file:
        # The 'files' parameter is used to upload the file
        files = {"file": (image_path, image_file, "image/jpeg")}
        
        # Send the POST request
        response = requests.post(url, files=files)
        
        # Print the server's response
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")

except FileNotFoundError:
    print(f"Error: The file was not found at {image_path}")
except Exception as e:
    print(f"An error occurred: {e}")
