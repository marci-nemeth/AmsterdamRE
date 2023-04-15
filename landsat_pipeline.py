import hashlib
import hmac
import base64
import urllib.parse as urlparse
import requests
import cv2
import numpy as np


def green_space(img_path,circular_mask= False):
    try:
      
        img = cv2.imread(img_path)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]

        if circular_mask:
            mask = np.zeros((height, width), np.uint8)
            cv2.circle(mask, (width//2, height//2), min(height, width)//2, (255, 255, 255), -1)
            img = cv2.bitwise_and(img, img, mask=mask)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img,(36, 25, 25), (70, 255,255))
        
        result = cv2.bitwise_and(img, img, mask=mask)

        
        radius = min(height, width) // 2


        area = np.pi * (radius ** 2)
        white_pix = cv2.countNonZero(mask)



        tot_pix = mask.size
    
        ratio = white_pix/area
        return ratio
    except Exception as e:
        print(str(e))
        return np.nan
    
    
def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
    """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()


def generate_map(coords,fname,zoom=19,dims = [512,512],verbose=True):
    """Generates a map of the area around the coordinates given
    Args:
        coords (list): [lat,lon] of the center of the map
        fname (str): name of the file to save the map to
        zoom (int, optional): zoom level of the map. Defaults to 19.
        dims (list, optional): dimensions of the map. Defaults to [512,512].
        verbose (bool, optional): print status messages. Defaults to True.
    Returns:
        None
    """
    
    api_key = ""
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    secret = ""
    signature = ""
    FINAL_URL = url + "center=" + str(coords[1])+","+str(coords[0])+ "&zoom=" +str(zoom) + "&size="+str(dims[0])+"x"+str(dims[1])+"&maptype=satellite"+"&key=" +api_key 
    input_url = "".join(FINAL_URL)
    signed_url =  sign_url(input_url, secret)
    r = requests.get(signed_url)
    with open('data//landsat//{}.png'.format(fname), 'wb') as f:
        f.write(r.content)
    if verbose:
        print("Map saved to data//landsat//{}.png".format(fname))



    