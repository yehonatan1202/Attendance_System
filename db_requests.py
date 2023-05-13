import numpy as np
import requests
import json

# Define the URL
url = 'http://yehonatan1202.pythonanywhere.com/'

# Check if the RFID is valid
def valid_rfid(rfid):
    r = requests.get(f'{url}/valid_rfid/{rfid}')
    return True if r.text != "None" else False

# Get the vector for a given RFID
def get_vector(rfid):
    # json -> list -> ndarray
    r = requests.get(f'{url}/get_vector/{rfid}')
    return np.array(json.loads(r.text))

# Get the content for a given RFID
def get_content(rfid):
    r = requests.get(f'{url}/get_content/{rfid}')
    return r.text

# Mark user with given rfid as present
def present(rfid):
    requests.get(f'{url}/present_rfid/{rfid}')

# Set vector for given student name
def set_vector(content, vector):
    # ndarry -> list -> json
    string_vector = json.dumps(vector.tolist())
    r = requests.get(
        f'{url}/set_vector/{content}/{string_vector}')
