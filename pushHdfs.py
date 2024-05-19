import os
from jsonTool import inputJson


def pushHdfsByName(dataDir, imgName):
    '''
    Push a single image
    '''
    print(dataDir, imgName, 'ok')


def pushHdfs(dataDir='./data/temp/data_maps_202211'):
    '''
    Push all images
    '''
    mapIndexTargetFile = os.path.join(dataDir, 'mapIndexTarget.json')
    mapObj = inputJson(mapIndexTargetFile)
    for map in mapObj:
        name = map.get('name')
        print('name:', name)
        isInHdfs = map.get('isInHdfs')
        if isInHdfs: continue
        print('will pushHdfs name:', name)
        pushHdfsByName(dataDir, name)


def testCutOne():
    # cutOneBigImg(res, '49SGU')
    dir = './data/temp/data_maps_202211'
    pushHdfsByName(dir, '50RPV')


if __name__ == '__main__':
    # testCutOne()
    # cutAllImg()
    # testCutOneCoordinates()
    pass
