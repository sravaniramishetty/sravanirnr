import pandas as pd
import numpy as np

df = pd.read_csv("ratings.csv")
df.to_csv("new_null.csv",encoding='utf-8')
#print(dfnull)
dfframe = pd.DataFrame()
for row in range(len(df.index)):
	df1 = df.loc[row]
	df1.to_csv("row1.csv")
	dfnew = pd.read_csv("row1.csv")
	person_name = dfnew.columns[1]
	dfnew.rename(columns={list(dfnew)[1]: "Transactions"},inplace = True)
	dfnew.insert(0 ,"name",person_name)
	
	total = dfnew.iloc[435]["Transactions"]
	dfnew = dfnew.dropna(how = 'any' , axis =0)
	for i in range(len(dfnew.index)):
		dfnew.iat[i,2] = dfnew.iloc[i,2]/total
		
	dfnew = dfnew.drop(dfnew.index[len(dfnew.index)-1])
	if len(dfnew.index)<15:
		dfnew = pd.DataFrame()
	else:
		dfnew = dfnew.drop(dfnew.index[(len(dfnew.index)-(len(dfnew.index)-)):])
	dfframe= dfframe.append(dfnew,ignore_index = True)

dfframe.to_csv("base.csv",encoding='utf-8',index=False)




	#total = dfnew.iloc[435]["Transactions"]
	#for i in range(len(dfnew.index)):
	#	dfnew.iat[i,2] = dfnew.iloc[i,2]/total
	#dfnew = dfnew.drop(dfnew.index[len(dfnew.index)-1])
#dfnew.to_csv("row1.csv")
	#df_final= dfframe.append(dfnew,ignore_index = True)
#dfnew = dfnew.drop(dfnew.index[len(dfnew.index)-1])
#df_final.to_csv("final.csv")
#print(dfnew)