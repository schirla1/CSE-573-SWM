Fraud detection using graph database
# Data
Consist of Dataset <br>
BankSim Dataset : bs140513_032310.csv <br>
Under Sampled Data : sampled_dataset.csv <br>
Graph based Data : graph_features_dataset.csv <br>

# Code
DatasetSampling.py - Generates under sampled file dataset.csv  <br>
<br>
GraphDatasetCreation.py- Generates graph features from neo4j to newDataset2.csv <br>

GraphBuildNeo4j.cyp - Commands for pushing the data into neo4j tool.

Features Extracted : merchDegree ,custDegree , merchCloseness , custCloseness , custPageRank , merchPageRank , custBetweeness ,merchBetweeness ,merchlouvain , custlouvain ,merchCommunity ,custCommunit <br>

StandardModel.py - Trains  raw features (sampled_dataset.csv) on RandomForestClassifier, classify_with_kmeans  and computes feature importance. <br>

GraphModel.py - Trains  graph features (graph_features_dataset.csv) on RandomForestClassifier, classify_with_kmeans  and computes feature importance.




