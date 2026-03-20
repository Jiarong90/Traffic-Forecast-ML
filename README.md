**End results:**

Trained 2 different ML pipelines to forecast T+15 traffic in Singapore

Pipeline 1 - one model predicts all - roads with changing speedbands and roads with static speedbands  

The best results achieved was 64% accuracy with 12% recall on actual changed rows, over 62% persistence baseline

Pipeline 2 - 3-stage pipeline - 3 task is decomposed into 3 stages. 
1. Gatekeeper - predicts if changes will occur. Recall is 80%, precision 50%
2. Router - predicts the direction of the change. 80% accuracy
3. Specialist models - Ascent Specialist and Descent Specialist. Predicts the magnitude of change for specified direction.

**Results from this pipeline:**
Results show that, the 3-stage pipeline was able to predict over 50% of overall larger traffic events where speedbands changed by 3 and above. 

The ascent specialist was able to predict 50% of larger recoveries, with about 64% precision. 

The descent specialist was able to predict 52.7% of larger traffic jams, with about 64% precision. On even larger jams (4+ speedband drop), the descent specialist was able to predict 59.9% of the jams with 74% precision. 

Furthermore, the MAE is about 0.56, meaning that the predicted bands are off by a magnitude of 0.56. The within1 is 0.96, showing that of the predicted speedbands, 96% of them are within 1 band of real value. 

The recall of changed rows is 22%, beating the original model by about 45%. This recall is selective. We selected a lower threshold for "gatekeeper" to achieve higher recall to catch all real changes, but it was a trade-off resulting in lower precision at 50%. Because of this, the specialist models were also trained on a balanced dataset, so it is able to further find signals in the data to detect true positives or false positives. 

The specialist showed that for lower speedband changes, there was low separation in data, with about 0.45 for 0 and 0.61 confidence for 1. Hence, these samples could include noise, which may not necessary be informative to the user. 
Hence, we introduced a threshold in the specialist too, to filter out samples at the cost of recall, but ensuring that confidence is moderate when alerting users of predictions.

Work done to achieve this pipeline

**Data polling**
Polling script datapolling.py was created, designing the sqlite3 database schemas to align with LTA Datamall's data structure. 
Then, polling script was run continuously every 5mins between 7am-10pm everyday to collect LTA/NEA data. 
APIs polled: LTA TrafficSpeedBands, TrafficIncidents, EstTravelTimes, FaultyTrafficLights, VMS/EMAS, TrainServiceAlerts, NEA Rainfall

**Data Export**
Data was exported to parquet format to help with querying. Because of size of data, querying for ML purposes is not possible with Sqlite3.

**Feature cleaning/engineering**
- Incidents snapshots mapped to one incident lifecycle
- Incidents mapped to Road Links
- Mapped neighbors affected by incidents
- Mapped Weather Stations to Road Links
- Created weather features

**Feature Table**
Built using DuckDB and combining parquet files on 21-day window. Fuse all data to be prepared for ML training.

**Training setup**
Chronological 80/20 to prevent data leakage

**Model Setup**
Gatekeeper - binary:logistic y_tp15 != sb
Router - binary:logistic y_tp15 < sb
Descent Specialist - reg:squarederror max(sb - y_tp15, 0)
Ascent Specialist - reg:squarederror max(y_tp15 - sb, 0)
