from PIL import Image
import os
from dirTool import findIMGDataByName
from jsonTool import inputJson, outputJson

Image.MAX_IMAGE_PIXELS = 2300000000
count = 40


def splitLine(s, e, count):
    res = []
    offset = (s - e) / count
    for i in range(0, count + 1):
        res.append(s - offset * i)
    return res


def splitListLine(lineStart, lineEnd, count):
    res = []
    for i in range(0, count + 1):
        sx = lineStart[i]
        ex = lineEnd[i]
        res.append(splitLine(sx, ex, count))
    return res


def getJsonIndex(i, j):
    return str(i) + "_" + str(j)


def cutOneCoordinates(coordinates):
    """
    Cut the range of one map sheet, cut in order by column, corresponding to the cut map sheet
    Parameters:
      coordinates - 2D array, length 4, representing the four corner coordinates of the map sheet, the second dimension length is 2
    Returns: Cut coordinates
    """
    cs = {}
    leftUpPoint = coordinates[0]
    rightUpPoint = coordinates[1]
    rightBottomPoint = coordinates[2]
    leftBottomPoint = coordinates[3]

    leftColumnX = splitLine(leftUpPoint[0], leftBottomPoint[0], count)
    upRowY = splitLine(leftUpPoint[1], rightUpPoint[1], count)
    rightColumnX = splitLine(rightUpPoint[0], rightBottomPoint[0], count)
    bottomRowY = splitLine(leftBottomPoint[1], rightBottomPoint[1], count)
    rowX = splitListLine(leftColumnX, rightColumnX, count)
    columnY = splitListLine(upRowY, bottomRowY, count)

    for i in range(0, count):
        sx = rowX[i]
        ex = rowX[i + 1]
        for j in range(0, count):
            name = getJsonIndex(j, i)
            sy = columnY[j]
            ey = columnY[j + 1]
            leftUp = [sx[j], sy[i]]
            rightUp = [sx[j + 1], ey[i]]
            rightBottom = [ex[j + 1], ey[i + 1]]
            leftBottom = [ex[j], sy[i + 1]]
            cs[name] = {"c": [leftUp, rightUp, rightBottom, leftBottom]}
    return cs


# Calculate the center point coordinates as the image name based on the coordinate range
def getSmallImgName(coordinates, index):
    img = coordinates[index]
    print('img:', index, img)
    coordinate = img['c']
    leftUpPoint = coordinate[0]
    rightUpPoint = coordinate[1]
    rightBottomPoint = coordinate[2]
    leftBottomPoint = coordinate[3]
    sX = min(leftUpPoint[0], rightUpPoint[0], rightBottomPoint[0], leftBottomPoint[0])
    eX = max(leftUpPoint[0], rightUpPoint[0], rightBottomPoint[0], leftBottomPoint[0])
    centerX = sX + (eX - sX) / 2
    sY = min(leftUpPoint[1], rightUpPoint[1], rightBottomPoint[1], leftBottomPoint[1])
    eY = max(leftUpPoint[1], rightUpPoint[1], rightBottomPoint[1], leftBottomPoint[1])
    centerY = sY + (eY - sY) / 2
    return str(centerX) + "_" + str(centerY)


def cutOneBigImg(bigImgFilePath, coordinates, mapName, smallImgDir):
    '''
    Only cut the image
    '''
    img = Image.open(bigImgFilePath)
    w, h = img.size
    print(w, h)
    sumW = 0
    for i in range(0, count):
        stepW = 274
        if (i % 2 == 0):
            stepW = 275
        newSumW = sumW + stepW
        sumH = 0
        for j in range(0, count):
            index = getJsonIndex(i, j)
            imgName = getSmallImgName(coordinates, index)
            imgName = mapName + "_" + index + "_" + imgName
            print('imgName:', imgName)
            stepH = 274
            if (j % 2 == 0):
                stepH = 275
            newSumH = sumH + stepH
            rect = (sumW, sumH, newSumW, newSumH)
            print(rect)
            temp = img.crop(rect)
            name = os.path.join(smallImgDir, imgName + '.jpg')
            print(name)
            temp.save(name)
            sumH = newSumH
        sumW = newSumW


def findMapPolygonByName(mapIndexTargetFile, mapName):
    mapObj = inputJson(mapIndexTargetFile)
    for m in mapObj:
        if (m['name'] == mapName): return m.get('polygon')
    return None


def setMapCut(mapIndexTargetFile, mapName):
    mapObj = inputJson(mapIndexTargetFile)
    for m in mapObj:
        if (m['name'] == mapName):
            m['isCut'] = True
    outputJson(mapObj, mapIndexTargetFile)


def generateCutJson(cutFile):
    '''
    Check if the file exists, if not, generate an empty json file
    '''
    if (os.path.exists(cutFile)): return
    outputJson({}, cutFile)


def cutImgByName(dataDir, imgName, level='L1C', onlyCoordinate=False):
    '''
    Cut one image and coordinate range
    '''
    if not onlyCoordinate:
        rsiDir = os.path.join(dataDir, 'rsi')
        print('rsiDir:', rsiDir)
        imgPath = findIMGDataByName(rsiDir, imgName, level)
        print('imgPath:', imgPath)
        if not imgPath: return
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    range = findMapPolygonByName(mapIndexTargetFile, imgName)
    print('range:', range)
    cutResFile = os.path.join(dataDir, 'cut.json')
    generateCutJson(cutResFile)
    cutObj = inputJson(cutResFile)
    mapObj = cutObj.get(imgName)
    if (not mapObj):
        mapObj = cutOneCoordinates(range)
        cutObj[imgName] = mapObj
        outputJson(cutObj, cutResFile)
    if not onlyCoordinate:
        smallImgDir = os.path.join(dataDir, 'rsi_small', imgName)
        if not os.path.exists(smallImgDir):
            os.makedirs(smallImgDir)
        cutOneBigImg(imgPath, mapObj, imgName, smallImgDir)
    setMapCut(mapIndexTargetFile, imgName)


def cutAllImg(dataDir='./data/temp/data_maps_202211', level='L1C', onlyCoordinate=False):
    '''
    Cut all images
    '''
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapIndexTargetFile)
    for map in mapObj:
        name = map.get('name')
        print('name:', name)
        if (not onlyCoordinate):
            download = map.get('download')
            if (not download): continue
        isCut = map.get('isCut')
        if isCut: continue
        print('will cut name:', name)
        cutImgByName(dataDir, name, level, onlyCoordinate)


def testCutOne():
    dir = './data/temp/data_maps_202211'
    cutImgByName(dir, '50RPV')


def testCutOneCoordinates():
    dataDir = './data/temp'
    imgName = '50SPA'
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    range = findMapPolygonByName(mapIndexTargetFile, imgName)
    res = cutOneCoordinates(range)
    print('ok', res)


if __name__ == '__main__':
    pass
