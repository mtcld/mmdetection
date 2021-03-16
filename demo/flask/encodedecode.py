import base64
import json
import base64
from base64 import encodebytes, decodebytes

'''
This script is created to encode the image and then use the json for post method on the API
'''

# # Encoding
image = open('sample.jpg', 'rb')
image_read = image.read()
image_64_encode = encodebytes(image_read)
ready_to_send = image_64_encode.decode("utf-8")

# # JSON
ready_to_send_json = {
    "data": ready_to_send
}

# # Saving JSON
with open("data.json", "w") as write_file:
    json.dump(ready_to_send_json, write_file)

# # Decoding - (Avaialble on API)
# byte_ready = bytes(ready_to_send, 'utf-8')
# image_64_decode = decodebytes(byte_ready) 
# image_result = open('b.jpg', 'wb') # create a writable image and write the decoding result
# image_result.write(image_64_decode)