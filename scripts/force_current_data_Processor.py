import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

#define function to calculate adjusted r-squared
def adjR(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = 1- (((1-(ssreg/sstot))*(len(y)-1))/(len(y)-degree-1))

    return results


def df_normalize(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def findslope(xi,xf,yi,yf):
    slope=(yf-yi)/(xf-xi)
    return slope

#Low pass filter
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
# Sample Rate
samplerate=100
# Filter requirements.
order = 6
fs = 42      # sample rate, Hz
cutoff = 4  # desired cutoff frequency of the filter, Hz


plt.close('all')


#For loop to make a list of all data names (Could probably make a dictionary so they're variables not strings)
file_list=[]
for i in range(5,11):
    j=str(i)
    file_list.append('Data_'+'6'+'_'+j+'00G'+'.csv')

    
#For loop to load all files into the python script (Doesnt use for loop bc strings not variables for now)

data0=pd.read_csv(file_list[0])
data1=pd.read_csv(file_list[1])
data2=pd.read_csv(file_list[2])
data3=pd.read_csv(file_list[3])
data4=pd.read_csv(file_list[4])
data5=pd.read_csv(file_list[5])

# Create Columns for mass (g)
mass_500_list=[500]*len(data0)
mass_600_list=[600]*len(data0)
mass_700_list=[700]*len(data0)
mass_800_list=[800]*len(data0)
mass_900_list=[900]*len(data0)
mass_1000_list=[1000]*len(data0)

#Append column to each data frame
data0['mass']=mass_500_list
data1['mass']=mass_600_list
data2['mass']=mass_700_list
data3['mass']=mass_800_list
data4['mass']=mass_900_list
data5['mass']=mass_1000_list

# Create Column for datarun number
run_0_list=[0]*len(data0)

#Append column to each data frame
data0['run']=run_0_list
data1['run']=run_0_list
data2['run']=run_0_list
data3['run']=run_0_list
data4['run']=run_0_list
data5['run']=run_0_list

# Creat column for filtered current ACS
ACS_Isens_filtered_0 = butter_lowpass_filter(data0['ACS_Isens'], cutoff, fs, order)
ACS_Isens_filtered_1 = butter_lowpass_filter(data1['ACS_Isens'], cutoff, fs, order)
ACS_Isens_filtered_2 = butter_lowpass_filter(data2['ACS_Isens'], cutoff, fs, order)
ACS_Isens_filtered_3 = butter_lowpass_filter(data3['ACS_Isens'], cutoff, fs, order)
ACS_Isens_filtered_4 = butter_lowpass_filter(data4['ACS_Isens'], cutoff, fs, order)
ACS_Isens_filtered_5 = butter_lowpass_filter(data5['ACS_Isens'], cutoff, fs, order)

#Append column to each data frame
data0['ACS_Isens_Filtered']=ACS_Isens_filtered_0
data1['ACS_Isens_Filtered']=ACS_Isens_filtered_1
data2['ACS_Isens_Filtered']=ACS_Isens_filtered_2
data3['ACS_Isens_Filtered']=ACS_Isens_filtered_3
data4['ACS_Isens_Filtered']=ACS_Isens_filtered_4
data5['ACS_Isens_Filtered']=ACS_Isens_filtered_5

# Create column for filtered force
Fsens_filtered_0 = butter_lowpass_filter(data0['Fsens'], cutoff, fs, order)
Fsens_filtered_1 = butter_lowpass_filter(data1['Fsens'], cutoff, fs, order)
Fsens_filtered_2 = butter_lowpass_filter(data2['Fsens'], cutoff, fs, order)
Fsens_filtered_3 = butter_lowpass_filter(data3['Fsens'], cutoff, fs, order)
Fsens_filtered_4 = butter_lowpass_filter(data4['Fsens'], cutoff, fs, order)
Fsens_filtered_5 = butter_lowpass_filter(data5['Fsens'], cutoff, fs, order)

#Append column to each data frame
data0['Fsens_Filtered']=Fsens_filtered_0
data1['Fsens_Filtered']=Fsens_filtered_1
data2['Fsens_Filtered']=Fsens_filtered_2
data3['Fsens_Filtered']=Fsens_filtered_3
data4['Fsens_Filtered']=Fsens_filtered_4
data5['Fsens_Filtered']=Fsens_filtered_5










#Plot current data for each run in ACS 
# fig, ax1 = plt.subplots()
# plt.title("ACS")
# data=data5
# time=data['Time']
# ACS_Isens=data['ACS_Isens']
# ACS_Isens_Filtered=data['ACS_Isens_Filtered']

# ax1.plot(time, ACS_Isens,label="Data"+str(i+5)+"00")
# ax1.plot(time, ACS_Isens_Filtered,label="Data"+str(i+5)+"00")

# ax1.legend(loc="upper right")


# fig, ax1 = plt.subplots()
# plt.title("ACS")
# for i in range(0,6): 
#     data=pd.read_csv(file_list[i])
#     time=data['Time']
#     ACS_Isens=data['ACS_Isens']

    
#     ax1.plot(time, ACS_Isens,label="Data"+str(i+5)+"00")


# ax1.legend(loc="upper right")

# #Plot current  data for each run in BH10
# fig, ax1 = plt.subplots()
# plt.title("BH10")
# for i in range(0,6): 
#     data=pd.read_csv(file_list[i])
#     time=data['Time']
#     BH_10_Isens=data['BH_10_Isens']
#     ax1.plot(time, BH_10_Isens,label="Data"+str(i+5)+"00")

# ax1.legend(loc="upper right")


# #Plot force data for each run 
# fig, ax1 = plt.subplots()
# plt.title("Fsens")
# for i in range(0,6): 
#     data=pd.read_csv(file_list[i])
#     time=data['Time']
#     Fsens=data['Fsens']
#     ax1.plot(time, Fsens,label="Data"+str(i+5)+"00")

# ax1.legend(loc="upper right")



#Inspect plots to see start time
#Trim data by "x" most likely 6 seconds
#Define sample rate: 10 samples in how many seconds then 10 seconds/delta time

samplerate=10/(data0['Time'].iloc[9]-data0['Time'].iloc[0])
trimseconds=6

data0=data0[round(trimseconds*samplerate):len(data0)]
data1=data1[round(trimseconds*samplerate):len(data1)]
data2=data2[round(trimseconds*samplerate):len(data2)]
data3=data3[round(trimseconds*samplerate):len(data3)]
data4=data4[round(trimseconds*samplerate):len(data4)]
data5=data5[round(trimseconds*samplerate):len(data5)]



#Concatenate all data for 1 set of runs
frames = [data0, data1, data2, data3,data4,data5]
set_data_1=pd.concat(frames)

#Define data points
ACS_Isens_filtered=set_data_1['ACS_Isens_Filtered']
Fsens_filtered=set_data_1['Fsens_Filtered']

# #Scatter plot of all data 
fig, ax1 = plt.subplots()
ax1.scatter(ACS_Isens_filtered, Fsens_filtered, label="Unfiltered", color = 'r',s=5)


#Fit polynomial line
model1 = np.poly1d(np.polyfit(ACS_Isens_filtered, Fsens_filtered, 1))
#Set x range for line min and max ranges +- 2.5%
min_x=ACS_Isens_filtered.min()-ACS_Isens_filtered.max()*.025
max_x=ACS_Isens_filtered.max()+ACS_Isens_filtered.max()*.025

polyline = np.linspace(min_x, max_x, 100)


#add fitted polynomial lines to scatterplot 
plt.plot(polyline, model1(polyline), color='green')

#calculated adjusted R-squared of each model
R1=adjR(ACS_Isens_filtered, Fsens_filtered, 1)

miny=Fsens_filtered.min()
maxy=Fsens_filtered.max()

ax1.text(max_x-max_x*0.1,miny+miny*.2, "R^"+str(round(R1['r_squared'],2)), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
ax1.set_xlabel("Current (Amps)")
ax1.set_ylabel("Force (grams)")
ax1.legend(loc="upper right")
ax1.set_ylim(miny*.9, maxy*1.1)
plt.grid()

#####################Color Coded Clusters General Plot of all data combined######################
fig, ax1 = plt.subplots()


datalength=len(data0)
i=0
color_list=['lawngreen','green','cyan','red','orange','purple']
mass_list=['500 g','600 g','700 g','800 g','900 g','1000 g']

for i in range(0,6):
    ax1.scatter(ACS_Isens_filtered.iloc[i*datalength:datalength*(i+1)], Fsens_filtered.iloc[i*datalength:datalength*(i+1)], color = color_list[i],s=5,label=mass_list[i])

ax1.set_xlabel("Current (Amps)")
ax1.set_ylabel("Force (grams)")
ax1.legend(loc="upper right")
ax1.set_ylim(miny*.9, maxy*1.1)


# #####################Average and STD of data (one run) and plot one point ######################


i=0
for i in range(0,6):
    current_data=ACS_Isens_filtered.iloc[i*datalength:datalength*(i+1)]
    force_data=Fsens_filtered.iloc[i*datalength:datalength*(i+1)]
    current_mean=current_data.mean()
    force_mean=force_data.mean()
    current_std=current_data.std()
    force_std=force_data.std()
    ax1.errorbar(current_mean,force_mean, xerr = current_std, yerr = force_std  ,fmt ='s',color='black')
ax1.legend(loc="lower right")                 
plt.title("Force vs Current")   






