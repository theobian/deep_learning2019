import pandas as pd
# from pandas.tools.plotting import table
import matplotlib.pyplot as plt


def make_df(label, data):
	df = pd.DataFrame()
	for i in range(len(label)):
		df.append(label[i], data[i])
	return df

def plot_scatter(df, x_ax, y_ax):
	df.plot(kind='scatter',x=x_ax,y=y_ax,color='red')
	plt.show()



def plot_line(df, x_ax, y_ax):
	ax = plt.gca()
	df.plot(kind='line',x=x_ax,y=y_ax, ax=ax)
	df.plot(kind='line',x=x_ax,y=y_ax, color='red', ax=ax)
	plt.show()



df_sub = pd.DataFrame({
	'model':['model 1', 'model 2', 'model 3', 'model 4', 'model 5'],
	'time':[107.2527208328247, 112.64579749107361, 114.64750266075134, 109.50039768218994, 67.97173953056335],
	'error train':[24.0, 6.2, 30.2, 23.8, 45.7],
	'error test':[24.5, 7.5, 27.4, 22.9, 42.8]
	})
# eta 0.1
# epochs 100
# batch size 20
# n_train 1000
# n_test 1000

