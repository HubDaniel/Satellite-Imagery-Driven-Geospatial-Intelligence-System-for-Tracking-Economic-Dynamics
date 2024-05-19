# Used to generate street view target point index for downloading street views
from downloadTarget import isAllOk
from cut import cutAllImg
from cutForScene import setAllPointsIndex
from streetview2 import downloadAllImage


def do():
    # Preparation work
    # 1. Create a blank directory
    # 2. Copy the mapIndexTarget.json file to the blank directory, modifications can be made
    # 3. Confirm the month
    dataDir = "./data/temp/data_scene02"

    # Step 1: Crop the map sheet range
    # Automatically create an empty cut.json file
    cutAllImg(dataDir, onlyCoordinate=True)
    if not isAllOk(dataDir, 'isCut'):
        print('Not all crops are complete')
        return

    # Step 2: Create the street view point index file scenePointsIndex.json
    # First, copy a target point file targetPoints2.csv to the directory
    # Output the cut.json file
    csvFileName = 'targetPoints2.csv'
    setAllPointsIndex(dataDir, csvFileName)

    # Step 3: Execute the download
    # To execute only the third step: run python data/streetview2.py, first confirm if the directory inside is correct
    # By checking the download field in cut.json, determine if the street views under a certain map sheet have all been downloaded
    downloadAllImage(dataDir)

    # The final output is a cut.json file
    # Structure:
    # {
    # "50SPA": { map sheet number
    #     "0_0": { cropped small map sheet index
    #         "c": [] # coordinates of the small map sheet range
    #         "sp": [ # street view points under the small map sheet
    #            {
    #                "p": [x, y] # coordinates of the street view point
    #                "d": 1  # street view point status, whether it has been downloaded


if __name__ == '__main__':
    do()
    pass
