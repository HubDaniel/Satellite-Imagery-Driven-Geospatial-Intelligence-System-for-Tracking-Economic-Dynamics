# Satellite-Imagery-Driven-Geospatial-Intelligence-System-for-Tracking-Economic-Dynamics

1. Download street view data: Run sceneFlow.py
2. Download satellite images, cut, calculate, summarize: Run processFlow2.py (non-spark) OR processFlowSpark.py (spark mode)
3. Model result prediction code: multi_eval2.py (non-spark), automatically called by processFlow2.py, can also be executed independently
4. multiEvalHdfs.py (spark mode), needs to be executed on a spark server
5. calcMultiScoreSpark.py (spark mode), calcMultiScore.py (non-spark)

Note: 1 and 2 are entry files with no specific order, but 2 depends on the data from 1.
