import warnings
import shutil
import requests
import re
import json
import time
import os
import logging
from sys import exit

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _search(keywords,page, max_results=None):
    url = 'https://duckduckgo.com/'
    params = {
    	'q': keywords
    }

    logger.debug("Hitting DuckDuckGo for Token")

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(url, data=params)
    searchObj = re.search(r'vqd=([\d-]+)\&', res.text, re.M|re.I)

    if not searchObj:
        logger.error("Token Parsing Failed !")
        return -1

    logger.debug("Obtained Token")

    headers = {
        'authority': 'duckduckgo.com',
        'accept': 'application/json, text/javascript, */* q=0.01',
        'sec-fetch-dest': 'empty',
        'x-requested-with': 'XMLHttpRequest',
        'user-agent': 'Mozilla/5.0 (Macintosh Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'cors',
        'referer': 'https://duckduckgo.com/',
        'accept-language': 'en-US,enq=0.9',
    }

    params = (
        ('l', 'us-en'),
        ('o', 'json'),
        ('q', keywords),
        ('vqd', searchObj.group(1)),
        ('f', ',,,'),
        ('p', f'{page}'),
        ('v7exp', 'a'),
    )

    requestUrl = url + "i.js"

    logger.debug("Hitting Url : %s", requestUrl)

    while True:
        while True:
            try:
                res = requests.get(requestUrl, headers=headers, params=params)
                data = json.loads(res.text)
                break
            except ValueError as e:
                logger.debug("Hitting Url Failure - Sleep and Retry: %s", requestUrl)
                time.sleep(5)
                continue

        logger.debug("Hitting Url Success : %s", requestUrl)
        # printJson(data["results"])

        if "next" not in data:
            logger.debug("No Next Page - Exiting")
            return data

        requestUrl = url + data["next"]

def printJson(objs):
    for obj in objs:
        print("Width {0}, Height {1}".format(obj["width"], obj["height"]))
        print("Thumbnail {0}".format(obj["thumbnail"]))
        print("Url {0}".format(obj["url"]))
        print("Title {0}".format(obj["title"].encode('utf-8')))
        print("Image {0}".format(obj["image"]))
        print("__________")

def search(query, start_ix, max_images):
    q = 0
    flag = 0
    iterator = 0
    data = list()
    if not os.path.exists(query):
        os.mkdir(query)
    while q<=max_images:
        logging.info(f"Downloading {iterator+1} set of images")
        res = _search(query,start_ix+iterator)
        for obj in res['results']:
            time.sleep(1)
            resp = requests.get(obj['image'], stream=True, verify=False)
            if resp.status_code == 200:
                with open(os.path.join(os.getcwd(),query, f'{q+1}.jpg'),'wb') as f:
                    shutil.copyfileobj(resp.raw, f)
                q+=1
            else:
                flag+=1
            if flag == 20:
                time.sleep(5)
                flag = 0
        iterator +=1
    logging.info("Downloading finished")
    return data
