# sentinelsat documentation https://sentinelsat.readthedocs.io/en/stable/api_overview.html
#
from sentinelsat import SentinelAPI, make_path_filter, InvalidChecksumError, ServerError
from datetime import date
from mathTool import getPolygonArea
from jsonTool import outputJson, inputJson
import os

# api = SentinelAPI('872480080@qq.com', '123456Abcdef*', 'https://apihub.copernicus.eu/apihub')
api = SentinelAPI('872480080@qq.com', '123456Abcdef*', 'https://catalogue.dataspace.copernicus.eu')

def saveFailProduct(directoryPath, productId):
    if isProductInFailList(directoryPath, productId): return
    #
    jsonFile = os.path.join(directoryPath, 'downloadFail.json')
    pList = inputJson(jsonFile)
    pList.append(productId)
    outputJson(pList, jsonFile)

def isProductInFailList(directoryPath, productId):
    jsonFile = os.path.join(directoryPath, 'downloadFail.json')
    pList = inputJson(jsonFile)
    for p in pList:
        if p == productId: return True
    return False

def createDownloadFailFile(directoryPath):
    jsonFile = os.path.join(directoryPath, 'downloadFail.json')
    if os.path.exists(jsonFile): return
    outputJson([], jsonFile)

def downloadPart(productId, filter, directoryPath):
    print('filter:', filter)
    path_filter = make_path_filter(filter)  # *_TCI.jp2 OR, "*_PVI.jp2"
    try:
        res = api.download(productId, nodefilter=path_filter, directory_path=directoryPath)
        print('download res:', res)
        nodes = res.get('nodes')
        keys = nodes.keys()
        if len(keys) >= 2: return True
        print('only download manifest.safe, no others')
        return False
    except InvalidChecksumError:
        print('productId:', productId, " download fail")
        return False
    except ServerError:
        print('productId:', productId, " download fail, ServerError")
        saveFailProduct(directoryPath, productId)
        return False

def downloadPartTCI(productId, directoryPath, level='L1C'):
    print('Downloading TCI.jp2')
    # return downloadPart(productId, '*_TCI.jp2', directoryPath)
    if level == 'L1C':
        return downloadPart(productId, '*_TCI.jp2', directoryPath)
    elif level == 'L2A':
        return downloadPart(productId, '*_TCI_*.jp2', directoryPath)

def downloadPartPVI(productId, directoryPath):
    print('Downloading PVI.jp2')
    return downloadPart(productId, '*_PVI.jp2', directoryPath)

def transFootprintToPolygon(footprint):
    # print('footprint1:', footprint)
    points = []
    # POLYGON ((118.06487986420781 32.532846626599884,119.23355267788659 32.517518690633835,119.20965184507907 31.52767425655353,118.05347599928325 31.54242728634164,118.06487986420781 32.532846626599884))

    if footprint.find('MULTIPOLYGON (((') >= 0:
        footprint = footprint.replace('MULTIPOLYGON (((', '')
        footprint = footprint.replace(')))', '')
        print('footprint:', footprint)
        points = footprint.split(', ')
    else:
        footprint = footprint.replace('POLYGON ((', '')
        footprint = footprint.replace('))', '')
        print('footprint:', footprint)
        points = footprint.split(',')
    print('points:', points)
    polygon = []
    for pstr in points:
        p = pstr.split(' ')
        polygon.append([float(p[0]), float(p[1])])
    # print('polygon:', polygon)
    return polygon

def calcArea(footprint):
    polygon = transFootprintToPolygon(footprint)
    polygon.pop()
    print('Number of points:', len(polygon))
    print('polygon:', polygon)
    area = getPolygonArea(polygon)
    return area

def downloadProduct(mapNum, dir=None, startDate=None, endDate=None, onlyPre=False, noDown=False, area=None, level='L1C'):
    """
    Main download method, first use query to search
    Parameter: onlyPre: true to only download preview images
    """
    if dir is None: dir = './data/temp/'
    if startDate is None: startDate = '20230101'
    if endDate is None: endDate = '20230130'
    createDownloadFailFile(dir)
    #
    filename = 'S2A_MSI' + level + '_*T' + mapNum + '_*'
    print('filename:', filename)
    # date = ('20221201', date(2023, 1, 11))
    products = api.query(date=(startDate, endDate),
                         platformname='Sentinel-2',
                         filename=filename,
                         cloudcoverpercentage=(0, 30)
                         )
    minCCPkey = ''
    minCCP = 100
    title = ''
    keys = products.keys()
    print('startDate:', startDate, ',endDate:', endDate)
    print('Found', len(keys), 'products')
    if len(keys) == 0: return None
    i = 0
    for key in keys:
        product = products[key]
        print('product:', i, product)
        if isProductInFailList(dir, key):
            print('Product', key, 'has been marked as undownloadable')
            continue
        i += 1
        ccp = product['cloudcoverpercentage']
        if area:
            footprint = product['footprint']
            mapArea = calcArea(footprint)
            rate = mapArea / area
            print('mapArea:', mapArea, ', ratio:', rate)
            # Only download if area is greater than 80% of the map area
            if mapArea < area * 0.8:
                print('Area too small, ratio', round(rate * 100, 2), '%')
                print('-------------end', i, 'product')
                continue
        print('ccp:', ccp)
        if minCCP > ccp and ccp > -1:
            minCCP = ccp
            minCCPkey = key
            title = product['title']
        print('key:', key)
        print('ccp:', ccp)
        # is_online = api.is_online(key)
        # print('is_online:', is_online)
        print('-------------end', i, 'product')

    print('minCCPkey:', minCCPkey)
    print('minCCP:', minCCP)
    if not minCCPkey:
        print('No suitable product id found')
        return None
    isOnline = api.is_online(minCCPkey)
    print('is_online:', isOnline)
    if not isOnline:
        print('Product', minCCPkey, 'is not online.')
        try:
            api.trigger_offline_retrieval(minCCPkey)
        except:
            print('Set', minCCPkey, 'online -- failed')
        else:
            print('Set', minCCPkey, 'online -- success')
        return None
    #
    if noDown: return None
    #
    if minCCPkey != '':
        print('Starting download----', mapNum, ',', minCCPkey)
        product = products[minCCPkey]
        print('title:', product['title'])
        r = False
        if onlyPre:
            r = downloadPartPVI(minCCPkey, dir)
        else:
            r = downloadPartTCI(minCCPkey, dir, level)
            downloadPartPVI(minCCPkey, dir)
        if r: return title
        else: print('An exception occurred during the download process')
    else:
        print('No suitable data found')
    print(mapNum, 'download execution ended')
    return None

# downloadProduct('50SKD')
# downloadProduct('49SGU')
# downloadProduct('50SLD')
# downloadProduct('49SFT')

def test1():
    dir = './data/temp/'
    isOK = downloadProduct('50SKD', dir)
    print('isOK:', isOK)

# Test image range
def testRange():
    downloadProduct('50SQA', startDate='20221201', endDate='20221231', noDown=True, area=1.149833482574195)

def test2():
    p = { 'title': 'S2A_MSIL1C_20221222T025131_N0509_R132_T50SQA_20221222T043346',
        'link': "https://apihub.copernicus.eu/apihub/odata/v1/Products('e6663737-a3c3-4207-a468-06cb9cc3ed0c')/$value",
        'link_alternative': "https://apihub.copernicus.eu/apihub/odata/v1/Products('e6663737-a3c3-4207-a468-06cb9cc3ed0c')/",
        'link_icon': "https://apihub.copernicus.eu/apihub/odata/v1/Products('e6663737-a3c3-4207-a468-06cb9cc3ed0c')/Products('Quicklook')/$value",
        'summary': 'Date: 2022-12-22T02:51:31.024Z, Instrument: MSI, Satellite: Sentinel-2, Size: 489.80 MB',
        'ondemand': 'false',
        #   'datatakesensingstart': datetime.datetime(2022, 12, 22, 2, 51, 31, 24000),
        #    'generationdate': datetime.datetime(2022, 12, 22, 4, 33, 46),
        #    'beginposition': datetime.datetime(2022, 12, 22, 2, 51, 31, 24000),
        #    'endposition': datetime.datetime(2022, 12, 22, 2, 51, 31, 24000),
        #     'ingestiondate': datetime.datetime(2022, 12, 22, 6, 42, 49, 442000),
            'orbitnumber': 39168, 'relativeorbitnumber': 132,
            'cloudcoverpercentage': 2.35126121208735,
            'sensoroperationalmode': 'INS-NOBS',
            'gmlfootprint': '<gml:Polygon srsName="http://www.opengis.net/gml/srs/epsg.xml#4326" xmlns:gml="http://www.opengis.net/gml">\n   <gml:outerBoundaryIs>\n      <gml:LinearRing>\n         <gml:coordinates>32.502471333501944,119.91041919153258 32.404714251931544,119.88095351982457 32.257470172667105,119.83671806098937 32.11022321002103,119.79267265770432 31.962978674064388,119.74882931752504 31.815650258368127,119.70494457788733 31.66842206209188,119.66144867213816 31.521079785803828,119.61779121942314 31.518695443424015,119.6170839349817 31.529422327835963,119.10607624353317 32.519334872088244,119.12885884704357 32.502471333501944,119.91041919153258</gml:coordinates>\n      </gml:LinearRing>\n   </gml:outerBoundaryIs>\n</gml:Polygon>',
            'footprint': 'MULTIPOLYGON (((119.6170839349817 31.518695443424015, 119.61779121942314 31.521079785803828, 119.66144867213816 31.66842206209188, 119.70494457788733 31.815650258368127, 119.74882931752504 31.962978674064388, 119.79267265770432 32.11022321002103, 119.83671806098937 32.257470172667105, 119.88095351982457 32.404714251931544, 119.91041919153258 32.502471333501944, 119.12885884704357 32.519334872088244, 119.10607624353317 31.529422327835963, 119.6170839349817 31.518695443424015)))',
            'level1cpdiidentifier': 'S2A_OPER_MSI_L1C_TL_2APS_20221222T043346_A039168_T50SQA_N05.09',
            'tileid': '50SQA', 'hv_order_tileid': 'SA50Q',
            'format': 'SAFE',
            'processingbaseline': '05.09',
            'platformname': 'Sentinel-2',
            'filename': 'S2A_MSIL1C_20221222T025131_N0509_R132_T50SQA_20221222T043346.SAFE',
            'instrumentname': 'Multi-Spectral Instrument',
            'instrumentshortname': 'MSI',
            'size': '489.80 MB',
            's2datatakeid': 'GS2A_20221222T025131_039168_N05.09',
            'producttype': 'S2MSI1C',
            'platformidentifier': '2015-028A',
            'orbitdirection': 'DESCENDING',
            'platformserialidentifier': 'Sentinel-2A',
            'processinglevel': 'Level-1C',
            'datastripidentifier': 'S2A_OPER_MSI_L1C_DS_2APS_20221222T043346_S20221222T025125_N05.09',
            'granuleidentifier': 'S2A_OPER_MSI_L1C_TL_2APS_20221222T043346_A039168_T50SQA_N05.09',
            'identifier': 'S2A_MSIL1C_20221222T025131_N0509_R132_T50SQA_20221222T043346',
            'uuid': 'e6663737-a3c3-4207-a468-06cb9cc3ed0c'}
    footprint = p.get('footprint')
    footprint = footprint.replace('MULTIPOLYGON (((', '')
    footprint = footprint.replace(')))', '')
    print('footprint:', footprint)
    points = footprint.split(', ')
    print(points)
    polygon = []
    for pstr in points:
        p = pstr.split(' ')
        # print('p:', p)
        # print('p0:', p[0], 'p1:', p[1])
        # print('p0:', float(p[0]), 'p1:', float(p[1]))
        polygon.append([float(p[0]), float(p[1])])
    print('polygon:', polygon)
    area = calcArea(polygon)
    print('area:', area)
    # print(re.search('\(.*\)', footprint).span())

def test3():
    title = 'S2A_MSIL1C_20221020T023731_N0400_R089_T50SQA_20221020T042658'
    products = api.query(date=('20221001', '20221130'),
                         platformname='Sentinel-2',
                         filename=title + "*",
                         cloudcoverpercentage=(0, 30)
                         )
    keys = products.keys()
    for key in keys:
        product = products[key]
        print('product:', product)
        print('----end one project')

if __name__ == '__main__':
    # testRange()
    # test2()
    test3()
    pass
