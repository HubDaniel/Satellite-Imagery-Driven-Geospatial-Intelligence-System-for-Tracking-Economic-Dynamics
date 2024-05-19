# Execute directory: the directory where the data directory is located, e.g., python data/calcScore.py
import os
import sys

# print('sys.path:', sys.path)
pathPwd = os.getcwd()
# print('pathPwd:', pathPwd)
# rootPath = os.path.join(pathPwd, '../')
rootPath = pathPwd
# print('rootPath:', rootPath)
sys.path.append(rootPath)
print('sys.path:', sys.path)

from ePre.multi_eval2 import extPre
from jsonTool import inputJson, outputJson
from dirTool import findSmallImgDir, findSmallImgPath

defaultScenePath = os.path.join('./data/temp/data_scene02', "default_scene.jpg") # Default image: ePre/sat_sce_pics/default.jpg, a small blue sky photo

# pathPwd = os.path.join(pathPwd, 'data')

# def getSmallMapIndex(smallMapName):
#     subStr = smallMapName.split('_')
#     return subStr[1] + '_' + subStr[2]

def getOnePreScore(satImgPath, secImgPath = defaultScenePath):
    if not os.path.isfile(satImgPath):
        print("Image file does not exist:", satImgPath)
        return 0
    if not os.path.isfile(secImgPath):
        print("Street view file does not exist:", secImgPath)
        secImgPath = defaultScenePath
    res = extPre(satImgPath, secImgPath)
    # print('res-score:', res)
    return res

def getSmallMapScore(sp, imgFilePath, secDir):
    """
    Calculate the score of one street view point for a small image's four street view images and return the maximum value
    """
    if sp is None:  # No street view point
        print('No street view point')
        return getOnePreScore(imgFilePath)
    maxScore = 0
    pitchs = '0'
    headings = ['0', '90', '180', '270']
    for point in sp:
        score = 0
        isDownload = point.get('d')
        if isDownload is None: # Not downloaded yet
            print('No street view point')
            score = getOnePreScore(imgFilePath)
            maxScore = max(maxScore, score)
        else:
            if isDownload != 1 and isDownload != 2: # 1: Complete download; 2: Incomplete download
                print('Street view data not downloaded', point)
                score = getOnePreScore(imgFilePath)
                point['s'] = score
                print('score', score)
                maxScore = max(maxScore, score)
            else:
                [x, y] = point.get('p')
                for h in headings:
                    secName = "%s_%s_%s_%s.png" % (str(x), str(y), h, pitchs)
                    secPath = os.path.join(secDir, secName)
                    score = getOnePreScore(imgFilePath, secPath)
                    maxScore = max(maxScore, score)
                    point['s_' + h] = score
                    print('score_', h, ":", score, '===', secName)
    print('maxScore:', maxScore)
    return maxScore

def setScoreByName(dataDir, mapName, secDir):
    jsonFileName = os.path.join(dataDir, 'cut.json')
    allCutJson = inputJson(jsonFileName)
    cutJson = allCutJson.get(mapName)
    if not cutJson:
        print('No cropped coordinate range data')
        return None
    # print('cutJson:', cutJson)
    smallImgDir = os.path.join(dataDir, 'rsi_small')
    imgDir = findSmallImgDir(smallImgDir, mapName) # Image directory
    secDir = os.path.join(secDir, mapName) # Street view directory
    print('imgDir:', imgDir)
    count = 0
    for smallMapName, smallMapValue in cutJson.items():
        print('=========Calculating:', mapName, smallMapName)
        if smallMapName == 'download': continue
        [row, col] = smallMapName.split('_')
        imgFilePath = findSmallImgPath(imgDir, row, col)
        sp = smallMapValue.get('sp')
        smallMapValue['s'] = getSmallMapScore(sp, imgFilePath, secDir)
        count += 1
        print('smallMapValue score:', smallMapValue['s'], '; count:', count)
        print('')
        if count % 400 == 0:
            outputJson(allCutJson, jsonFileName)
    outputJson(allCutJson, jsonFileName)
    return 1

def setAllScore(dataDir = './data/temp/data_maps_202211', secDir = './data/temp/data_scene02'):
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapIndexTargetFile)
    for map in mapObj:
        name = map.get('name')
        download = map.get('download')
        if not download: continue
        isCut = map.get('isCut')
        if not isCut: continue
        print('name:', name)
        isScore = map.get('isScore')
        if isScore: continue
        r = setScoreByName(dataDir, name, secDir)
        if r == 1:
            map['isScore'] = True
            outputJson(mapObj, mapIndexTargetFile)

def testSetAllScore():
    dataDir = './data/temp/data_maps_202301'
    secDir = './data/temp/data_scene02/sce_pics'
    setAllScore(dataDir, secDir)

if __name__ == '__main__':
    # setScore()
    testSetAllScore()
    pass
