import pandas as pd
import csv 

data = pd.read_csv("Data/bs140513_032310.csv")
csvreader = csv.reader(data)
header = next(csvreader)

fraud1 = data[data['fraud']==1]

fraud0 = data[data['fraud']==0]

connections= []
customers = list(set(fraud1['customer']))
i=0
for cust in customers  :
    connections.append(fraud0[fraud0['customer']==cust])
    
customer_related = pd.concat(connections)
connections= []
merchants = list(set(fraud1['merchant']))
i=0
for mer in merchants  :
    connections.append(fraud0[fraud0['merchant']==mer])
merchant_related = pd.concat(connections)

with open('Data/sampled_dataset.csv', 'w+', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(merchant_related)
    
