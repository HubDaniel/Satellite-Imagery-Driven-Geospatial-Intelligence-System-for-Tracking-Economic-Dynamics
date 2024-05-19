# Test getting files
import os
import re

def findIMGDataByName(dir, fileName, level='L1C'):
    ls = os.listdir(dir)
    for d in ls:
        # print('d:', d)  # d is the file name
        subPath = os.path.join(dir, d)
        # print('path:', subPath)
        if os.path.isdir(subPath):
            resPath = findIMGDataByName(subPath, fileName, level)
            if resPath: return resPath
        else:
            str = '.*' + fileName + '_.*_TCI.jp2$'
            if level == 'L2A':
                str = '.*' + fileName + '_.*_TCI_10m.jp2$'
            if re.match(str, d):
                # print('----------')
                return subPath
    return None

def findSmallImgDir(dir, imgDirName):
    ls = os.listdir(dir)
    for d in ls:
        if d == imgDirName:
            subPath = os.path.join(dir, d)
            if os.path.isdir(subPath):
                return subPath
    return None

def findSmallImgPath(dir, row, col):
    ls = os.listdir(dir)
    for f in ls:
        subStr = f.split('_')
        if str(row) == subStr[1] and str(col) == subStr[2]:
            return os.path.join(dir, f)
    return None

def findImgDir(dir, mapName):
    '''
    Determine if the map sheet with the given name has been downloaded
    '''
    ls = os.listdir(dir)
    for d in ls:
        subPath = os.path.join(dir, d)
        print('subPath:', subPath)
        str = '.*_T' + mapName + '_.*.SAFE$'
        if re.match(str, d):
            return subPath
    return None

def makeDir(dir):
    if os.path.exists(dir): return
    os.mkdir(dir)

def testFind():
    dir = './data/temp/data_maps_202211/rsi'
    t = findIMGDataByName(dir, '50RPV')
    print('ok', t)

def delDir(dir):
    '''
    First delete all files in the subdirectories, then delete the directories
    '''
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            # print('file:', os.path.join(root, name))
            os.remove(os.path.join(root, name))
        for name in dirs:
            # print('dir:', os.path.join(root, name))
            os.rmdir(os.path.join(root, name))
    os.rmdir(dir)

def testDel():
    dir = './data/temp/data_maps_20221202/rsi'
    delDir(dir)
    pass

def testFindSmallImgPath():
    dir = './data/temp/data_maps_202301/rsi_small/50SPA'
    path = findSmallImgPath(dir, 0, 3)
    print('path:', path)

if __name__ == '__main__':
    # testFind()
    # testDel()
    testFindSmallImgPath()
    pass
