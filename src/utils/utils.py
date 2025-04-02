from src.utils.configuration import Configuration
import shutil, os, getpass, socket, numpy as np, cv2, requests
from PIL import Image

def leq(a: float, b: float) -> bool:
    return a <= b

def geq(a: float, b: float) -> bool:
    return a >= b

def pil_to_cv2(pil_image):
    cv2_image = np.array(pil_image)  # Convert to NumPy array
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return cv2_image


def cv2_to_pil(cv2_image):
    #rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pil_image = Image.fromarray(cv2_image)  # Convert NumPy array to PIL image
    return pil_image


def adjust_coordinate_rectangle(points):
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, x_max, y_min, y_max

def send_ntfy_notification(topic):
    url = f"https://ntfy.sh/{topic}"
    username = getpass.getuser()
    hostname = socket.gethostname()
    project = "sud4vup"
    message = f"{project}: execution completed by {username} on {hostname}!!"
    response = requests.post(url, data=message.encode('utf-8'))
    if response.status_code == 200:
        print(f"Notification sent to topic '{topic}'.")
    else:
        print(f"Failed to send notification: {response.text}")

class FileCleaner():
    def __init__(self):
        self.folder_names = ['maskfolder', 'preprocessedfolder']

    def clean(self):
        configuration = Configuration()
        for folder_name in self.folder_names:
            folder = configuration.get(folder_name)
            shutil.rmtree(folder)
            os.makedirs(folder)


