# use os.system command to submit work

import os,time

def submitHdfs(type,satRoot,sceRoot,mapName,savePath):
    if(type=='local'):
        command = "/usr/local/Cellar/spark-3.4.1-bin-hadoop3/bin/spark-submit"
        # command += " --master spark://localhost:7077"
    else:
        command = "/usr/local/soft/spark-3.4.1-bin-hadoop3/bin/spark-submit"
        # command += " --master spark://cMaster:7077"
    # command += " --deploy-mode cluster"
    # command += " --executor-memory 4g"
    # command += " --num-executors 4"
    # command += " --py-files your_code.py"
    command += " "
    command += " /Users/qiaobin/tempProject/evisual_api/data/multiEvalHdfs.py"
    command += " "+satRoot
    command += " "+sceRoot
    command += " "+mapName
    command += " "+savePath
    #
    res=os.system(command)
    # res值
    # 0 成功
    # 512 can't open file '/Users/qiaobin/tempProject/evisual_api/data/multiEvalHdfs.py': [Errno 2] No such file or directory
    print('system-res:',res)
    return res
    
def generateName():
    return time.strftime("%Y_%m_%d_%H%M", time.localtime())

if __name__ == '__main__':
    # testToLocal()
    mapName='50RPU'
    savePath='/spark/result/'+mapName+"_"+generateName()
    submitHdfs('local','/spark/data_maps','/spark/data_scene',mapName,savePath)
