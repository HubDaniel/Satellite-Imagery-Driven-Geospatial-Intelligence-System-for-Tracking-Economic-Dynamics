# Execute directory: the directory where the data directory is located, e.g., python data/calcScore.py
import os
import sys
import time
# print('sys.path:', sys.path)
pathPwd = os.getcwd()
# print('pathPwd:', pathPwd)
# rootPath = os.path.join(pathPwd, '../')
rootPath = pathPwd
# print('rootPath:', rootPath)
sys.path.append(rootPath)
print('sys.path:', sys.path)

from jsonTool import inputJson, outputJson
from submitCmd import submitHdfs

def generateName():
    return time.strftime("%Y_%m_%d_%H%M", time.localtime())

def setScoreByName(dataDir, secDir, mapName):
    savePath = '/spark/result/' + mapName + '_' + generateName()
    res = submitHdfs('local', dataDir, secDir, mapName, savePath)
    if res == 0:
        return savePath
    return None

def setAllScore(dataDir = './data/temp/data_maps_202211', satDir = '', secDir = ''):
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapIndexTargetFile)
    for map in mapObj:
        name = map.get('name')
        download = map.get('download')
        if not download: continue
        isCut = map.get('isCut')
        if not isCut: continue
        isInHdfs = map.get('isInHdfs')
        if not isInHdfs: continue
        print('name:', name)
        hdfsScorePath = map.get('hdfsScorePath')
        if hdfsScorePath: continue
        resPath = setScoreByName(satDir, secDir, name)
        if resPath:
            map['hdfsScorePath'] = resPath
            outputJson(mapObj, mapIndexTargetFile)

def testSetAllScore():
    dataDir = './data/temp/data_maps_202301'
    secDir = './data/temp/data_scene02/sce_pics'
    setAllScore(dataDir, secDir)

if __name__ == '__main__':
    # setScore()
    testSetAllScore()
    pass
