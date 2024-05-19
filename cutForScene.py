# Crop the target map sheet to generate a small map index for street view image indexing

import os
from jsonTool import inputJson, outputJson, read_csv, write_csv
from mathTool import pointInPolygon
import copy


def setPointIndex(cutObj, mapName, point):
    """
    Traverse the grid of cutObj to find the small grid cell to which the point belongs
    """
    mapValue = cutObj.get(mapName)
    if mapValue is None:
        print("cut.json file is missing the map sheet: " + mapName)
        return
    for smallMapName, smallMapValue in mapValue.items():
        c = smallMapValue.get('c')
        if isPointInPolygon(point, c):
            sp = smallMapValue.get('sp')
            if sp is None:
                sp = []
                smallMapValue['sp'] = sp
            sp.append({'p': point})


def isPointInPolygon(point, polygon):
    plg = copy.deepcopy(polygon)
    plg.append(plg[0])
    return pointInPolygon(point, plg)


def getPointLocation(mapObj, point):
    """
    Get the mapName where the point is located
    """
    for map in mapObj:
        name = map.get('name')
        polygon = map.get('polygon')
        if isPointInPolygon(point, polygon):
            return name
    return None


def getFieldIndex(header, fieldName):
    for i in range(len(header)):
        if header[i] == fieldName: return i
    return -1


def getPoint(header, row):
    x = row[getFieldIndex(header, 'longitude')]
    x = float(x)
    y = row[getFieldIndex(header, 'latitude')]
    y = float(y)
    return [x, y]


def setAllPointsIndex(dataDir, csvFileName='testPoints.csv', batchCount=10000):
    """
    Parameters:
        batchCount: Save the file every batchCount number of points
    """
    # Read the contents of cut.json
    cutFile = os.path.join(dataDir, 'cut.json')
    cutObj = inputJson(cutFile)
    # Read mapIndexTarget.json
    mapFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapFile)
    # Read the point data to be classified points.csv
    csvFile = os.path.join(dataDir, csvFileName)
    data = read_csv(csvFile)
    header = data[0]
    header.append('isIndex')
    data = data[1:]
    # Traverse point data
    count = 0
    for i in range(len(data)):
        # Get point
        row = data[i]
        count += 1
        point = getPoint(header, row)
        # Check if the index has already been calculated
        isIndex = 0
        isIndexIndex = getFieldIndex(header, 'isIndex')
        if len(row) > isIndexIndex:
            isIndex = row[isIndexIndex]
        if isIndex == 0:  # Skip if index has already been added
            # Calculate the map sheet where the point is located and record it in the cut.json file
            mapName = getPointLocation(mapObj, point)
            print('mapName:', mapName, point, count)
            if mapName is not None:
                setPointIndex(cutObj, mapName, point)
            # Mark as indexed
            row.append(1)
        # Batch save
        if count % batchCount == 0:
            print('save_csv:', count, "================================")
            write_csv(csvFile, data, header)
            outputJson(cutObj, cutFile)

    write_csv(csvFile, data, header)
    outputJson(cutObj, cutFile)
    print('Index calculation completed')


def summary(dataDir):
    """
    Summarize the number of street view images in each map sheet and the total number in the cut.json file
    """
    cutFile = os.path.join(dataDir, 'cut.json')
    cutObj = inputJson(cutFile)
    sumFile = os.path.join(dataDir, 'cutSummary.json')
    sumObj = {}
    for mapName, mapValue in cutObj.items():
        mapObj = {"sum": 0, "sum_d1": 0, "sum_d2": 0, "sum_d3": 0, "sum_d4": 0}
        sumObj[mapName] = mapObj
        for smallMapName, smallMapValue in mapValue.items():
            if smallMapName == 'download': continue
            sp = smallMapValue.get('sp')
            if sp is None: continue
            spCount = len(sp)
            mapObj['sum'] += spCount
            for point in sp:
                d = point.get('d')
                if d is not None and d >= 1 and d <= 4:
                    mapObj['sum_d' + str(d)] += 1
            for index in range(4):
                strIndex = str(index + 1)
                mapObj['sum_rate_d' + strIndex] = str(round(mapObj['sum_d' + strIndex] / mapObj['sum'] * 100, 2)) + '%'

    print('sum:', sumObj)
    outputJson(sumObj, sumFile)


def testSetAllPointsIndex():
    dataDir = './data/temp/data_scene'
    setAllPointsIndex(dataDir, batchCount=5)


def testSetPointIndex():
    dataDir = './data/temp/data_scene'
    cutFile = os.path.join(dataDir, 'cut2.json')
    cutObj = inputJson(cutFile)
    setPointIndex(cutObj, '50SPA', [118.07, 32.51])
    print('cutObj:', cutObj)


def testGetPointLocation():
    dataDir = './data/temp/data_scene'
    mapFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapFile)
    name = getPointLocation(mapObj, [118.5, 32.0])
    print('name:', name)
    name = getPointLocation(mapObj, [18.5, 3.0])
    print('name2:', name)


def testGetFieldIndex():
    i = getFieldIndex(['a', 'b'], 'b')
    print('i:', i)
    i2 = getFieldIndex(['a', 'b'], 'c')
    print('i2:', i2)


def testSummary():
    dataDir = './data/temp/data_scene02'
    summary(dataDir)


if __name__ == '__main__':
    # testSetAllPointsIndex()
    # testSetPointIndex()
    # testGetPointLocation()
    # testGetFieldIndex()
    testSummary()
    pass
