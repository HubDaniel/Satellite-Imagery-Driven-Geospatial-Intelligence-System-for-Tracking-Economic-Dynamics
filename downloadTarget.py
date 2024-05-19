import os
from jsonTool import inputJson, outputJson
from downloadRSI import downloadProduct
from mathTool import getPolygonArea
from dirTool import makeDir, findImgDir, delDir

def downloadTargatMap(onlyPre=False, dataDir='./data/temp/data_maps_202211', startDate='20221001', endDate='20221130', level='L1C'):
    targetMapJsonfileDir = os.path.join(dataDir, 'mapIndexTarget.json')
    targetMapJson = inputJson(targetMapJsonfileDir)
    mapDir = os.path.join(dataDir, 'rsi')
    makeDir(mapDir)
    for map in targetMapJson:
        isDownload = map.get('download')
        area = map.get('area')
        mapName = map.get('name')
        print('map:', mapName, isDownload)
        if isDownload: continue
        d = findImgDir(mapDir, mapName)
        print('d:', d)
        if d:
            delDir(d)
            print('Deleted directory:', d)
        title = downloadProduct(map['name'], mapDir, startDate, endDate, onlyPre=onlyPre, area=area, level=level)
        if title:
            map['download'] = True
            map['title'] = title
            outputJson(targetMapJson, targetMapJsonfileDir)
        print('Map sheet', mapName, 'download process completed')

def calcTargatMapArea(dataDir='./data/temp/data_maps_202211'):
    targetMapJsonfileDir = os.path.join(dataDir, 'mapIndexTarget.json')
    targetMapJson = inputJson(targetMapJsonfileDir)
    for map in targetMapJson:
        polygon = map.get('polygon')
        area = getPolygonArea(polygon)
        print(map.get('name'), 'area:', area)
        map['area'] = area
    outputJson(targetMapJson, targetMapJsonfileDir)
    print('calcTargatMapArea ok')

def isAllOk(dataDir, fieldName):
    targetMapJsonfileDir = os.path.join(dataDir, 'mapIndexTarget.json')
    targetMapJson = inputJson(targetMapJsonfileDir)
    for map in targetMapJson:
        isOK = map.get(fieldName)
        if not isOK: return False
    return True

def isAllDownload(dataDir='./data/temp/data_maps_202211'):
    return isAllOk(dataDir, 'download')

if __name__ == '__main__':
    # downloadTargatMap(True)  # Download preview images
    # downloadTargatMap()  # Download all images
    # calcTargatMapArea()
    # allDownload = isAllDownload('./data/temp/data_maps_202207')
    # print('allDownload:', allDownload)
    pass
