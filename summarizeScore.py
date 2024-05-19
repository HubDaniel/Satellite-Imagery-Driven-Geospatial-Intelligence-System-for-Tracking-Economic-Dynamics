import os
import sys
# print('sys.path:',sys.path)
pathPwd = os.getcwd()
# print('pathPwd:',pathPwd)
# rootPath=os.path.join(pathPwd, '../')
rootPath = pathPwd
# print('rootPath:',rootPath)
sys.path.append(rootPath)
# sys.path.append('./data')
print('sys.path:', sys.path)

from jsonTool import inputJson, outputJson
from spark.hdfsTool import readSparkContent

def getTargetScore(scores, key):
    for score in scores:
        if score[0] == key:
            return score[1]
    return None

def summarizeScoreByName(dataDir, mapName, sparkFilePath):
    print('summarizeScoreByName:', mapName)
    jsonFileName = os.path.join(dataDir, 'cut.json')
    allCutJson = inputJson(jsonFileName)
    cutJson = allCutJson.get(mapName)
    if not cutJson:
        print(mapName, 'does not have cropped coordinate data')
        return None
    scores = readSparkContent(sparkFilePath)
    print('scores:', scores)
    for smallMapName, smallMapValue in cutJson.items():
        score = getTargetScore(scores, smallMapName)
        # print(mapName,' - ',smallMapName,'-score:',score)
        if score is not None:
            smallMapValue['s'] = score
    outputJson(allCutJson, jsonFileName)
    return 1

def summarizeAllScore(dataDir=''):
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapIndexTargetFile)
    for map in mapObj:
        name = map.get('name')
        print('name:', name)
        download = map.get('download')
        if not download:
            continue
        isCut = map.get('isCut')
        if not isCut:
            continue
        isInHdfs = map.get('isInHdfs')
        if not isInHdfs:
            continue
        hdfsScorePath = map.get('hdfsScorePath')
        if not hdfsScorePath:
            continue
        if hdfsScorePath == "None":
            continue
        isScore = map.get('isScore')
        if isScore:
            continue
        r = summarizeScoreByName(dataDir, name, hdfsScorePath)
        if r == 1:
            map['isScore'] = True
            outputJson(mapObj, mapIndexTargetFile)

if __name__ == '__main__':
    pass
