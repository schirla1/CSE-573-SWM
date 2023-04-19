# Import Required Modules
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
import csv

# Create Home Page Route
app = Flask(__name__)

#df = df.transpose()
@app.route('/')
def bar_with_plotly():
	import pandas as pd

	import matplotlib.pyplot as plt
	df = pd.read_csv("output2.csv")
	# data=['C1000148617', 'C100045114', 'C1000699316', 'C1001065306', 'C1002658784']
	# print(df.head())
	header = df.columns.values
	ans = df.loc[df.iloc[:,0] == "'C1000148617'"]

	cus_dict={}
	#print(ans.iloc[0])
	for i in range(1, len(header)):
		cus_dict[header[i]] = ans.iloc[0][i]
	print(cus_dict)
	courses = list(cus_dict.keys())
	values = list(cus_dict.values())
	fig = plt.figure(figsize = (20, 5))
	
	plt.bar(courses, values, color ='blue',
			width = 0.4)
	
	plt.xlabel("Customer Transaction Data")
	plt.ylabel("No. of transactions")
	plt.title("Type of transaction")
	plt.savefig('img.jpeg')
	

	# Use render_template to pass graphJSON to html
	return render_template('bar.html', data=data, imgpath='/img.jpeg')


if __name__ == '__main__':
	app.run(debug=True, port='8000')
