# Reference: https://blog.csdn.net/qq_25737169/article/details/113843872
# Gaussian Area Formula: https://zhuanlan.zhihu.com/p/612991648

def getPolygonArea(polygon):
    """
    Compute polygon area
    polygon: list with shape [n, 2], n is the number of polygon points
             The array should be sorted in vertex order, without overlapping the first and last points.
    """
    area = 0
    q = polygon[-1]
    # print('q:',q)
    for p in polygon:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area) / 2.0


def testArea1():
    polygon = [[0, 0], [-1, 1], [0, 2], [1, 1]]  # np.array([[0, 0], [-1, 1], [0, 2], [1, 1]]).astype("float32")
    area = getPolygonArea(polygon)
    print('area:', area, '=2')


def testArea2():
    polygon = [[1, 1], [1, 2], [2, 3], [3, 2], [2, 1]]
    area = getPolygonArea(polygon)
    print('area:', area, '=2.5')


def testArea3():
    polygon = [
        [120.2612310847352, 31.505173744758178],
        [120.29646450395221, 32.49414172713443],
        [119.21178795554155, 32.5175455307649],
        [119.20749733900473, 32.50130818639534],
        [119.16837807775894, 32.35345775779994],
        [119.12913230458496, 32.20558690909536],
        [119.12090879430653, 32.17390214187934],
        [119.10607624353317, 31.529422327835963],
        [120.2612310847352, 31.505173744758178]
    ]
    print(len(polygon))
    polygon.pop()
    print(len(polygon))
    area = getPolygonArea(polygon)
    print('area:', area)


# Reference: https://blog.csdn.net/tjcwt2011/article/details/102737081
# Principle: Ray casting method: Draw a ray from the target point and count the number of intersections with all edges of the polygon. If there are an odd number of intersections, it means inside, and if there are an even number, it means outside.
# Ray selection: Choose a line parallel to the x-axis

def pointInPolygon(point, polygon):
    """
    Determine if a point is inside a polygon
    Parameters:
        point: [x, y]
        polygon: [[x1, y1], [x2, y2], ...] must be a closed polygon
    """
    pLng = point[0]
    pLat = point[1]
    count = len(polygon)
    if count < 3:
        return False
    isIn = False
    j = count - 1
    for i in range(1, count):
        p1Lng, p1Lat = polygon[i]
        p2Lng, p2Lat = polygon[j]  # Last point
        # Choose a line parallel to the x-axis
        # First judge whether the point's line intersects with the line segment
        if ((p1Lat < pLat and p2Lat >= pLat) or (p2Lat < pLat and p1Lat >= pLat)):
            # Substitute the y value of the line into the line segment to get the x-coordinate of the intersection. If the x-value is less than the Point's x, it means the point is on the right side of the line segment. Just judge one side for an odd number.
            # Line equation, according to the same slope: (x-x1)/(y-y1)=(x2-x1)/(y2-y1) => x=(y-y1)*(x2-x1)/(y2-y1)+x1
            #   x1   +  ( y  -  y1 )  / ( y2   -   y1 ) * (  x2  -  x1 )
            if (p1Lng + (pLat - p1Lat) / (p2Lat - p1Lat) * (p2Lng - p1Lng) < pLng):
                isIn = not isIn
        j = i
    return isIn


def testPointInPolygon():
    point = [118.5, 32.0]
    polygon = [
        [118.0648798642, 32.5328466262],
        [119.2335526779, 32.5175186902],
        [119.2096518451, 31.52767425620001],
        [118.0534759993, 31.542427286]
    ]
    polygon.append(polygon[0])
    isIn = pointInPolygon(point, polygon)
    print('isIn', isIn)  # true


if __name__ == '__main__':
    testArea1()
    testArea2()
    testArea3()
    testPointInPolygon()
    pass
