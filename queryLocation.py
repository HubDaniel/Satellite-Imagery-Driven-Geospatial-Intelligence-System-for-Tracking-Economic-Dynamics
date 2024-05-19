import os
import sys

pathPwd = os.getcwd()
rootPath = pathPwd
sys.path.append(rootPath)
print('queryLocation-sys.path:', sys.path)

from data.jsonTool import inputJson, outputJson

def setOff(time, score):
    if time == '202301':
        score = score / 100
    else:
        score = score + 0.2
    return score

def queryCity(province, city, jsonDir, polygon = None):
    """
    Query or set city boundary coordinates. If no boundary coordinates (polygon) are passed, execute the query. If boundary coordinates are passed, set them.
    """
    jsonObj = inputJson(jsonDir)
    p = getProvince(province, jsonObj)
    c = p.get(city)
    if polygon is None:
        return c
    else:
        if c is not None:
            return c
        else:
            cityObj = {
                'provine': province,
                'city': city,
                'polygon': polygon
            }
            p[city] = cityObj
            outputJson(jsonObj, jsonDir)

def getProvince(province, jsonObj):
    p = jsonObj.get(province)
    if p is None:
        p = {}
        jsonObj[province] = p
    return p

def getCity(city, provinceObj):
    c = provinceObj.get(city)
    if c is None:
        c = {}
        provinceObj[city] = c
    return c

def getCenterPoint(coordinate):
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
    return [centerX, centerY]

# Reference document https://blog.csdn.net/tjcwt2011/article/details/102737081
# Judgment principle: Ray casting method: draw a ray from the target point and count the number of intersections with the polygon edges. If the number of intersections is odd, the point is inside; if even, it is outside.
# Ray selection: choose a ray parallel to the x-axis
def pointInPolygon(point, polygon):
    pLng = point[0]
    pLat = point[1]
    count = len(polygon)
    if count < 3: return False
    isIn = False
    j = count - 1
    for i in range(1, count):
        p1Lng, p1Lat = polygon[i]
        p2Lng, p2Lat = polygon[j]
        if ((p1Lat < pLat and p2Lat >= pLat) or (p2Lat < pLat and p1Lat >= pLat)):
            if (p1Lng + (pLat - p1Lat) / (p2Lat - p1Lat) * (p2Lng - p1Lng) < pLng):
                isIn = not isIn
        j = i
    return isIn

def queryCityMap(province, city):
    """
    Query the small maps contained in the city area. If they exist, return them directly; if not, calculate in real-time, save, and then return them.
    """
    jsonDir = './data/temp/location.json'
    locatonObj = inputJson(jsonDir)
    provinceObj = getProvince(province, locatonObj)
    cityObj = provinceObj.get(city)
    res = cityObj.get('smap')
    if res: return res
    print('Calculating map location')
    polygonStr = cityObj.get('polygon')
    plgs = polygonStr.split(';')
    polygon = []
    for pStr in plgs:
        p = pStr.split(',')
        lng = float(p[0])
        lat = float(p[1])
        polygon.append([lng, lat])
    cutJsnoDir = './data/temp/cut.json'
    cutObj = inputJson(cutJsnoDir)
    smap = []
    for key, values in cutObj.items():
        point = getCenterPoint(values.get('c'))
        if pointInPolygon(point, polygon):
            smap.append(values)
    cityObj['smap'] = smap
    outputJson(locatonObj, jsonDir)
    return smap

def queryDistrict(province, city, district, jsonDir, polygon):
    """
    Add province, city, district, and coordinate range to the JSON object. If they exist, return; if not, add them.
    """
    jsonObj = inputJson(jsonDir)
    p = getProvince(province, jsonObj)
    c = getCity(city, p)
    d = c.get(district)
    if d is not None: return d
    if polygon is None: return None
    else:
        districtObj = {
            'provine': province,
            'city': city,
            'district': district,
            'polygon': polygon
        }
        c[district] = districtObj
        outputJson(jsonObj, jsonDir)

def delDistrict(province, city, district, jsonDir):
    """
    Delete the district. If it exists, delete it, including all data under the district level such as coordinate ranges and contained small maps.
    """
    jsonObj = inputJson(jsonDir)
    p = jsonObj.get(province)
    if p is None: return True
    else:
        c = p.get(city)
        if c is None: return True
        else:
            d = c.get(district)
            if d is None: return True
            else:
                del c[district]
                outputJson(jsonObj, jsonDir)
                return True

def queryDistrictMap(dir, province, city, district, isReCalc = False):
    """
    Query the small maps contained in the district area. If they exist, return them directly; if not, calculate in real-time, save, and then return them.
    """
    jsonDir = os.path.join(dir, 'district.json')
    locatonObj = inputJson(jsonDir)
    provinceObj = getProvince(province, locatonObj)
    if provinceObj is None: return 'No such province'
    cityObj = provinceObj.get(city)
    if cityObj is None: return 'No such city'
    districtObj = cityObj.get(district)
    if districtObj is None: return 'No such district'
    res = districtObj.get('smap')
    if not isReCalc:
        if type(res) == list: # An empty array is also considered as data
            return res
    polygonStr = districtObj.get('polygon')
    plgs = polygonStr.split(';')
    polygon = []
    for pStr in plgs:
        p = pStr.split(',')
        lng = float(p[0])
        lat = float(p[1])
        polygon.append([lng, lat])
    cutJsnoDir = os.path.join(dir, 'cut.json')
    cutObj = inputJson(cutJsnoDir)
    smap = []
    for mapkey, mapValues in cutObj.items():
        for key, smallMap in mapValues.items():
            if type(smallMap) == type(True): continue
            point = getCenterPoint(smallMap.get('c'))
            smallMap['key'] = key
            smallMap['mapkey'] = mapkey
            if pointInPolygon(point, polygon):
                smap.append(smallMap)
    districtObj['smap'] = smap
    outputJson(locatonObj, jsonDir)
    return smap

def queryAllDistrictMap(dir):
    """
    Query all the small maps contained in all district areas without calculation
    """
    jsonDir = os.path.join(dir, 'district.json')
    locatonObj = inputJson(jsonDir)
    maps = []
    for province, provinceValues in locatonObj.items():
        for city, cityValues in provinceValues.items():
            for district, districtValues in cityValues.items():
                smap = districtValues.get('smap')
                if not smap: continue
                maps.extend(smap)
    return maps

def setAllDistrictMap(isReCalc = False, dataDir = './data/temp/data_maps_202211'):
    """
    Set the small maps within the range of all districts
    """
    districtFile = os.path.join(dataDir, 'district.json')
    if not os.path.exists(districtFile):
        print('No district coordinate file')
        return False
    districtObj = inputJson(districtFile)
    for province, provinceValues in districtObj.items():
        for city, cityValues in provinceValues.items():
            for district, districtValues in cityValues.items():
                polygon = districtValues.get('polygon')
                if not polygon: continue
                print('Setting small maps for:', province, city, district)
                queryDistrictMap(dataDir, province, city, district, isReCalc)
    return True

def getAvgScore(smap, isUsm):
    sum = 0
    i = 0
    for map in smap:
        s = map.get('s')
        if not s: continue
        i += 1
        sum = sum + s
    if i == 0: return 0
    if isUsm:
        return sum
    return sum / i

def setAllDistrictAvgScore(dataDir = './data/temp/data_maps_202211'):
    """
    Calculate the average score/sum of all districts
    """
    districtFile = os.path.join(dataDir, 'district.json')
    if not os.path.exists(districtFile):
        print('No district coordinate file')
        return False
    districtObj = inputJson(districtFile)
    isSum = True  # Calculate the sum
    for province, provinceValues in districtObj.items():
        for city, cityValues in provinceValues.items():
            for district, districtValues in cityValues.items():
                polygon = districtValues.get('polygon')
                if not polygon: continue
                smap = districtValues.get('smap')
                if not smap: continue # Execute continue when smap is an empty array []
                score = getAvgScore(smap, isSum)
                print('Calculating:', province, city, district, 'Total score:' if isSum else 'Average score:', score)
                districtValues['sumScore' if isSum else 'avgScore'] = score
    outputJson(districtObj, districtFile)

def queryAllDistrictAvgScore(dataDir = './data/temp/data_maps_202211', dataYear = None):
    """
    Query the average score of all districts
    """
    districtFile = os.path.join(dataDir, 'district.json')
    districtObj = inputJson(districtFile)
    scores = []
    for province, provinceValues in districtObj.items():
        for city, cityValues in provinceValues.items():
            for district, districtValues in cityValues.items():
                polygon = districtValues.get('polygon')
                if not polygon: continue
                avgScore = districtValues.get('avgScore')
                if not avgScore: continue
                avgScore = setOff(dataYear, avgScore)
                scores.append({
                    "province": province,
                    "city": city,
                    "district": district,
                    "avgScore": avgScore,
                    "polygon": polygon
                })
    return scores

def queryAllDistrictYearbook(districtFile):
    """
    Query the yearbook GDP of all districts for a specific year
    """
    districtObj = inputJson(districtFile)
    gdp = []
    for province, provinceValues in districtObj.items():
        for city, cityValues in provinceValues.items():
            for district, districtValues in cityValues.items():
                polygon = districtValues.get('polygon')
                if not polygon: continue
                avgScore = districtValues.get('totalValue')
                if not avgScore: continue
                gdp.append({
                    "province": province,
                    "city": city,
                    "district": district,
                    "avgScore": avgScore,
                    "polygon": polygon
                })
    return gdp

def queryDistrictScore(dataDir, province, city, district):
    """
    Query the score of a specific district
    """
    jsonDir = os.path.join(dataDir, 'district.json')
    if not os.path.exists(jsonDir): return
    jsonObj = inputJson(jsonDir)
    p = jsonObj.get(province)
    if p is None: return
    else:
        c = p.get(city)
        if c is None: return
        else:
            d = c.get(district)
            if d is None: return
            else:
                avgScore = d.get('avgScore')
                return avgScore

def getMapsTime(mapdir = ""):
    dir = mapdir.split('_')
    return dir[len(dir) - 1]

def queryDistrictMonthScore(dataDirs = ['./data/temp/data_maps_202211'], province = '', city = '', district = ''):
    """
    Query the data of a specific district for several months
    """
    scores = []
    for dataDir in dataDirs:
        score = queryDistrictScore(dataDir, province, city, district)
        dirTime = getMapsTime(dataDir)
        score = setOff(dirTime, score)
        scores.append({dirTime: score})
    return scores

if __name__ == '__main__':
    s=queryDistrictMonthScore(['./data/temp/data_maps_202213'],'安徽省','芜湖市','南陵县')
    print(s)
    print('ok')
