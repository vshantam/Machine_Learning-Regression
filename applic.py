#importing Libraries
from lib import *

def main():
	try:
		#setting API key for unlimited Access to Free data sets from Quandl;
		quandl.ApiConfig.api_key ='8NCm_V55725o9WHs9wsb'#add your key;
	
		#extracting the dataset 
		data=quandl.get_table('WIKI/PRICES')#or choose your tables
	
		#Using only those which are important
		data=data[['adj_open','adj_high','adj_low','adj_close','adj_volume','date']];
		#calculating High-LOw Percentage for the data for each row
		data['hl_pct']=(data['adj_high']-data['adj_close'])/data['adj_close'] *100;
		data['pct_chg']=(data['adj_high']-data['adj_open'])/data['adj_open'] *100;
		#setting those column to main dataframe
		data=data[['adj_open','adj_close','adj_high','adj_low','adj_volume','hl_pct','pct_chg']];
		
		forcast_lable='adj_close'
		#filling unavailable data with 'NaN'
		data.fillna(-99999,inplace=True)
		#shifiting the data
		forcast_out=int(math.ceil(0.01*len(data)))
		
		#creating lables for the shifted data
		data['lable']=data[forcast_lable].shift(+forcast_out)
		data.dropna(inplace=True)
		
		#converting datas to the numpy array
		x=np.array(data.drop(['lable'],1))
		#scaling the data
		x=preprocessing.scale(x)
		y=np.array(data['lable'])
		
		#displaying the length of rows and columns
		print(len(x),len(y))
		
		#Training and Testing the data using cros_Validaton algorithm
		x_test,xtrain,y_test,ytrain=cross_validation.train_test_split(x,y,test_size=0.2)
		
		#Using linear regression algorithm for fitting and creating the classifier
		clf=LinearRegression()
		clf.fit(xtrain,ytrain)
		
		#calculating the accuracy
		accuracy=clf.score(x_test,y_test)
		print(accuracy)
		
		#Making predictions using Linear regression algorithms
		x=x[:-forcast_out]
		x_pred=x[-forcast_out:]
		forcast_set=clf.predict(x_pred)
		print (forcast_set);
		
		#creating graph for the real and shifted stock data
		data=data[['adj_open','adj_close','adj_high','adj_low','adj_volume','hl_pct','pct_chg','lable']];
		data['adj_close'].plot()
		data['lable'].plot()
		#creating Lables and locating
		plt.legend(loc=1)
		plt.show()
	except Exception as e:
		print("Error occured:{}".format(str(e)))

			
if __name__=='__main__':
	main()
