# Multimodal workflow
from downloadTarget import downloadTargatMap, calcTargatMapArea, isAllOk
from cut import cutAllImg
from calcMultiScore import setAllScore
from queryLocation import setAllDistrictMap, setAllDistrictAvgScore

def do():
    # Preparation work
    # 1. Create a blank directory
    # 2. Copy the mapIndexTarget.json file to the blank directory, modifications can be made
    # 3. The rsi directory does not need to be created, it can be generated automatically
    # 4. Confirm the month
    dataDir = "./data/temp/data_maps_202301"
    secDir = './data/temp/data_scene02/sce_pics'
    startDate = '20230101'
    endDate = '20230130'
    level = 'L2A'

    # Step 1: Image download
    calcTargatMapArea(dataDir)  # Calculate area
    downloadTargatMap(dataDir=dataDir, startDate=startDate, endDate=endDate, onlyPre=False, level=level)  # Download data
    if not isAllOk(dataDir, 'download'):
        print('Not all downloads are complete')
        return

    # Step 2: Image cropping
    # Automatically create an empty cut.json file
    cutAllImg(dataDir, level)
    if not isAllOk(dataDir, 'isCut'):
        print('Not all crops are complete')
        return

    # Step 3: Enter the model and calculate the score
    # The cut.json generated in step 2 is not used, copy a cut.json from data_scene02, which is output by sceneFlow.py and contains the coordinates of street views
    setAllScore(dataDir, secDir)
    if not isAllOk(dataDir, 'isScore'):
        print('Not all score calculations are complete')
        return

    # Step 4: District summary
    # Manually copy a district boundary district.json file
    setAllDistrictMap(dataDir=dataDir, isReCalc=True)  # Find small maps for each district
    setAllDistrictAvgScore(dataDir)  # Calculate the average score for each district
    pass

if __name__ == '__main__':
    do()
    pass
