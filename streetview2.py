# Improved download
import re, os
import json
import requests
import time, glob
import csv
import sys
from jsonTool import write_csv, read_csv, inputJson, outputJson
from dirTool import makeDir

def grab_img_baidu(_url, _headers=None):
    if _headers is None:
        # Set request header
        headers = {
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
            "Referer": "https://map.baidu.com/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
        }
    else:
        headers = _headers
    response = requests.get(_url, headers=headers)

    if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
        return response.content
    else:
        return None

def openUrl(_url):
    # Set request header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    response = requests.get(_url, headers=headers)
    if response.status_code == 200:  # If status code is 200, the server has successfully processed the request, continue processing data
        return response.content
    else:
        return None

def getPanoId(_lng, _lat):
    # Get svid of baidu streetview
    url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day&t=1530956939770" % (str(_lng), str(_lat))
    response = openUrl(url).decode("utf8")
    if response is None:
        return None
    reg = r'"id":"(.+?)",'
    pat = re.compile(reg)
    try:
        svid = re.findall(pat, response)[0]
        return svid
    except:
        return None

# Official conversion function
# Because baidu streetview uses doubly encrypted baidu Mercator projection bd09mc (Change wgs84 to baidu09)
def wgs2bd09mc(wgs_x, wgs_y):
    # to:5 converts to bd0911, 6 converts to baidu Mercator
    url = 'http://api.map.baidu.com/geoconv/v1/?coords={}&from=1&to=6&output=json&ak={}'.format(
        wgs_x + ',' + wgs_y,
        'mYL7zDrHfcb0ziXBqhBOcqFefrbRUnuq'
    )
    res = openUrl(url).decode()
    temp = json.loads(res)
    bd09mc_x = 0
    bd09mc_y = 0
    if temp['status'] == 0:
        bd09mc_x = temp['result'][0]['x']
        bd09mc_y = temp['result'][0]['y']

    return bd09mc_x, bd09mc_y

def downloadOneImage(svid, h, imgPath):
    try:
        url = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={}&heading={}&pitch=0&width=1024&height=512'.format(svid, h)
        img = grab_img_baidu(url)
        if img is None: return False
        with open(imgPath, "wb") as f:
            f.write(img)
        return True
    except Exception as e:
        return False

def downloadOnePointImage(p, dir, filenames_exist=[]):
    """
    Download four street views of a point
    Return value: -1: local file already exists; 0: no street view; 1: complete download; 2: incomplete download; 3: coordinate to Mercator conversion error; 4: getPanoId error
    """
    headings = ['0', '90', '180', '270']  # directions, 0 is north
    pitchs = '0'
    wgs_x, wgs_y = str(p[0]), str(p[1])
    # Check if file already exists, skip if one exists
    flag = True
    for k in range(len(headings)):
        flag = "%s_%s_%s_%s.png" % (wgs_x, wgs_y, headings[k], pitchs) in filenames_exist
        if flag: break
    if flag:
        print('file exist:', p)
        return -1
    # Coordinate to Mercator conversion
    try:
        bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
    except requests.exceptions.ConnectionError as e:
        print('Connection error, program will exit. e:', e)
        sys.exit()
    except Exception as e:
        print('wgs2bd09mc-error:', str(e))
        return 3
    try:
        svid = getPanoId(bd09mc_x, bd09mc_y)
    except Exception as e:
        print('getPanoId-error:', e)
    if svid is None:
        print('not find panoId', p)
        return 4
    summary = 0
    for hi in range(len(headings)):
        h = headings[hi]
        imgPath = os.path.join(dir) + '/%s_%s_%s_%s.png' % (wgs_x, wgs_y, h, pitchs)
        downloadOneImageIsOK = downloadOneImage(svid, h, imgPath)
        if downloadOneImageIsOK:
            summary += 1
    if summary == 0: return 0
    elif summary == 4: return 1
    else: return 2

def downloadImages(sp, dir, filenames_exist):
    """
    Download street views for multiple points
    Parameter: dir, directory to save images, directory for a map sheet
    """
    count = 0  # Currently, a maximum of 2 points can be successfully downloaded for a small image
    index = 0
    isDo = False
    for point in sp:
        index += 1
        print('start download:', point, ';index:', index)
        d = point.get('d')
        if d is not None:  # Skip if already downloaded
            print('has exist', point)
            count += 1
            if count >= 2:  # Maximum of 2 points can be successfully downloaded for a small image
                break
            continue  # None: not downloaded; 0: no street view; 1: complete download; 2: incomplete download
        # Execute a download
        p = point.get('p')
        res = downloadOnePointImage(p, dir, filenames_exist)
        if res == -1:
            point['d'] = 1
            count += 1
            if count >= 2:  # Maximum of 2 points can be successfully downloaded for a small image
                break
            continue  # File already exists, directly assign 1, skip
        if res == 1:
            print('Download successful:', point)
            count += 1
        else:
            print('end download: ', res)
        point['d'] = res
        # Remember to sleep for 6s, too fast may result in being blocked
        print('downloaded, start==========================sleep 6')
        isDo = True
        time.sleep(6)
        if count >= 2:  # Maximum of 2 points can be successfully downloaded for a small image
            break
    return isDo

def downloadAllImage(dataDir, cutFile='cut.json'):
    sce_pics = os.path.join(dataDir, 'sce_pics')
    makeDir(sce_pics)
    cutFile = os.path.join(dataDir, cutFile)
    cutObj = inputJson(cutFile)
    count = 1
    for mapName, mapValue in cutObj.items():
        download = mapValue.get('download')
        if download:
            print(mapName, 'all downloaded')
            continue  # Skip if all downloaded
        mapPath = os.path.join(sce_pics, mapName)
        makeDir(mapPath)
        filenames_exist = glob.glob1(mapPath, "*.png")
        for smallMapName, smallMapValue in mapValue.items():
            if type(smallMapValue) == type(True): continue  # Skip if not of dictionary type
            sp = smallMapValue.get('sp')
            if sp is None: continue
            print(mapName, '_', smallMapName, ":", sp, ";", len(sp))
            print('count for save:', count, '==')
            isDo = downloadImages(sp, mapPath, filenames_exist)
            if isDo: count += 1  # Only increment if work was done
            # Save once
            if count % 5 == 0:
                outputJson(cutObj, cutFile)
        mapValue['download'] = True
        outputJson(cutObj, cutFile)

def testDownloadAllImage():
    dataDir = './data/temp/data_scene02'
    downloadAllImage(dataDir)
    pass

def testDownloadOnePointImage():
    dataDir = './data/temp/data_scene02/t1'
    downloadOnePointImage([120.134432, 30.195597], dataDir)

if __name__ == "__main__":
    testDownloadAllImage()
    # testDownloadOnePointImage()
    # return
