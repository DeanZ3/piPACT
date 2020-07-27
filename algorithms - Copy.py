from dataAnalysis import *
import math
from math import e
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit

#Basic linear regression line
"""
x = np.log(getAllDistances())
y = getAllRSSI("Short_A")
plt.plot(x, y, 'o')

m, b = np.polyfit(x, y, 1)

plt.plot(x, m*x + b)
plt.show()
"""




"""
***************************************************************************************/
*    Title: Datatofish source code
*    Author: Data to Fish
*    Date: 4/3/2020
*    Code version: Modified
*    Availability: https://datatofish.com/multiple-linear-regression-python/
*
***************************************************************************************/
"""


#Original method to create DataFrame with pandas
#Different length lists for each variation, so put aside
"""
Bluetooth = {'ORSSI': getAllRSSI("No_Obstructions"),
                '2SRSSI': getAllRSSI("2_Shorts"),
                '2JRSSI': getAllRSSI("2_Jeans"),
                'SRSSI': getAllRSSI("Shelf_M"),
                'HRSSI': getAllRSSI("Human_A"),

                'ODistance': np.log(getAllDistances("No_Obstructions")),
                '2SDistance': np.log(getAllDistances("2_Shorts")),
                '2JDistance': np.log(getAllDistances("2_Jeans")),
                'SDistance': np.log(getAllDistances("Shelf_M")),
                'HDistance': np.log(getAllDistances("Human_A")),

                'ORealDistance': getAllDistances("No_Obstructions"),
                '2SRealDistance': getAllDistances("2_Shorts"),
                '2JRealDistance': getAllDistances("2_Jeans"),
                'SRealDistance': getAllDistances("Shelf_M"),
                'HRealDistance': getAllDistances("Human_A")

                #'Temperature': np.log(getAllTemp("Short_A")),
                #'Humidity': np.log(getAllHumidity("Short_A")),
                }

headers = ['ORSSI', '2SRSSI', '2JRSSI', #'SRSSI', 'HRSSI',
    'ODistance', '2SDistance', '2JDistance', #'SDistance', 'HDistance',
    'ORealDistance', '2SRealDistance', '2JRealDistance'] #'SRealDistance', 'HRealDistance']

#df = pd.DataFrame(Bluetooth,columns=['RSSI','Distance','Temperature','Humidity'])
df = pd.DataFrame(Bluetooth, columns = headers)
"""

#Create DataFrame that holds all data via pandas for algorithms

#O = No_Obstructions
#2S = 2_Shorts
#2J = 2_Jeans
#S = Shelf_M
#H = Human_A
l1 = getAllRSSI("No_Obstructions")
l2 = getAllRSSI("2_Shorts")
l3 = getAllRSSI("2_Jeans")
l4 = getAllRSSI("Shelf_M")
l5 = getAllRSSI("Human_A")

l6 = np.log(getAllDistances("No_Obstructions"))
l7 = np.log(getAllDistances("2_Shorts"))
l8 = np.log(getAllDistances("2_Jeans"))
l9 = np.log(getAllDistances("Shelf_M"))
l10 = np.log(getAllDistances("Human_A"))

l11 = getAllDistances("No_Obstructions")
l12 = getAllDistances("2_Shorts")
l13 = getAllDistances("2_Jeans")
l14 = getAllDistances("Shelf_M")
l15 = getAllDistances("Human_A")

l16 = np.log(getAllTemp("No_Obstructions"))
l17 = np.log(getAllTemp("2_Shorts"))
l18 = np.log(getAllTemp("2_Jeans"))
l19 = np.log(getAllTemp("Shelf_M"))
l20 = np.log(getAllTemp("Human_A"))

l21 = np.log(getAllHumidity("No_Obstructions"))
l22 = np.log(getAllHumidity("2_Shorts"))
l23 = np.log(getAllHumidity("2_Jeans"))
l24 = np.log(getAllHumidity("Shelf_M"))
l25 = np.log(getAllHumidity("Human_A"))


s1 = pd.Series(l1, name = 'ORSSI')
s2 = pd.Series(l2, name = '2SRSSI')
s3 = pd.Series(l3, name = '2JRSSI')
s4 = pd.Series(l4, name = 'SRSSI')
s5 = pd.Series(l5, name = 'HRSSI')

s6 = pd.Series(l6, name = 'ODistance')
s7 = pd.Series(l7, name = '2SDistance')
s8 = pd.Series(l8, name = '2JDistance')
s9 = pd.Series(l9, name = 'SDistance')
s10 = pd.Series(l10, name = 'HDistance')

s11 = pd.Series(l11, name = 'ORealDistance')
s12 = pd.Series(l12, name = '2SRealDistance')
s13 = pd.Series(l13, name = '2JRealDistance')
s14 = pd.Series(l14, name = 'SRealDistance')
s15 = pd.Series(l15, name = 'HRealDistance')

s16 = pd.Series(l16, name = 'OTemperature')
s17 = pd.Series(l17, name = '2STemperature')
s18 = pd.Series(l18, name = '2JTemperature')
s19 = pd.Series(l19, name = 'STemperature')
s20 = pd.Series(l20, name = 'HTemperature')

s21 = pd.Series(l21, name = 'OHumidity')
s22 = pd.Series(l21, name = '2SHumidity')
s23 = pd.Series(l21, name = '2JHumidity')
s24 = pd.Series(l21, name = 'SHumidity')
s25 = pd.Series(l21, name = 'HHumidity')

df = pd.concat([s1, s2, s3, s4, s5,
                s6, s7, s8, s9, s10,
                s11, s12, s13, s14, s15,
                s16, s17, s18, s19, s20,
                s21, s22, s23, s24, s25], axis = 1)
df.fillna(df.mean(), inplace=True)

print(df)



#Create models of X and Y variations
#One X is for RSSI, temperature and humidity, and the other is just RSSI


#XO = df[['ODistance', 'OTemperature', 'OHumidity']]
XO = df[['ODistance']]
YO = df['ORSSI']
# with sklearn
O = linear_model.LinearRegression()
O.fit(XO, YO)

#X2S = df[['2SDistance', '2STemperature', '2SHumidity']]
X2S = df[['2SDistance']]
Y2S = df['2SRSSI']
S2 = linear_model.LinearRegression()
S2.fit(X2S, Y2S)

#X2J = df[['2JDistance', '2JTemperature', '2JHumidity']]
X2J = df[['2JDistance']]
Y2J = df['2JRSSI']
J2 = linear_model.LinearRegression()
J2.fit(X2J, Y2J)


#XS = df[['SDistance', 'STemperature', 'SHumidity']]
XS = df[['SDistance']]
YS = df['SRSSI']
S = linear_model.LinearRegression()
S.fit(XS, YS)

#XH = df[['HDistance', 'HTemperature', 'HHumidity']]
XH = df[['HDistance']]
YH = df['HRSSI']
H = linear_model.LinearRegression()
H.fit(XH, YH)

#Quick regression line constants prinout
print('Intercept: \n', O.intercept_)
print('Coefficients: \n', O.coef_)
print('Intercept: \n', S2.intercept_)
print('Coefficients: \n', S2.coef_)
print('Intercept: \n', J2.intercept_)
print('Coefficients: \n', J2.coef_)
print('Intercept: \n', S.intercept_)
print('Coefficients: \n', S.coef_)
print('Intercept: \n', H.intercept_)
print('Coefficients: \n', H.coef_)

# Print header stuff to get inputted data
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

# with sklearn
#Right now it is O for No_Obstructions
Intercept_result = ('Intercept: ', O.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)
Coefficients_result  = ('Coefficients: ', O.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# New RSSI label and input box
label1 = tk.Label(root, text='Type RSSI: ')
canvas1.create_window(100, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New Temperature label and input box
label2 = tk.Label(root, text=' Type Temperature: ')
canvas1.create_window(120, 120, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# New Humidity Label and input box
label3 = tk.Label(root, text='Type Humidity: ')
canvas1.create_window(114, 140, window=label3)
entry3 = tk.Entry (root) # create 3rd entry box
canvas1.create_window(270, 140, window=entry3)


# Calculate prediction w/ Tempertaure & Humidity
#Right now O is for No_Obstructions
def values():
    global RSSI #our 1st input variable
    RSSI = np.float32(entry1.get())

    global Temperature #our 2nd input variable
    Temperature = np.float32(entry2.get())
    Temperature = np.log(Temperature)

    global Humidity #our 3rd input variable
    Humidity = np.float32(entry3.get())
    Humidity = np.log(Humidity)


    prediction = RSSI - np.float32(O.intercept_)

    #Comment out these 2 lines for when not include temperature & humidity
    #prediction -= (O.coef_[1] * Temperature)
    #prediction -= (O.coef_[2] * Humidity)

    prediction /= O.coef_[0]
    prediction = pow(e, prediction)
    Prediction_result  = ('Predicted Distance: ', prediction)
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)

#Show "predict distancne" button
button1 = tk.Button(root, text='Predict Distance',command=values, bg='orange') # button to call the 'values' command above
canvas1.create_window(270, 170, window=button1)


#Function to test folder w/ model for t/f positives/negatives
def test(folderName):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    for file in getAllFolderFiles(folderName):
        actual = getDistance(file)
        RSSI = avgRSSI(getRSSIData(folderName, file))
        Temperature = np.log(getTemp(file))
        Humidity = np.log(getHumidity(file))
        prediction = RSSI - np.float32(H.intercept_)

        #Comment out these 2 when temperatrue and humidity are not included
        #prediction -= (H.coef_[1] * Temperature)
        #prediction -= (H.coef_[2] * Humidity)

        prediction /= H.coef_[0]
        prediction = pow(e, prediction)
        if truePositive(actual, prediction):
            trueP += 1
        if falsePositive(actual, prediction):
            falseP += 1
        if trueNegative(actual, prediction):
            trueN += 1
        if falseNegative(actual, prediction):
            falseN += 1
        print(getDistance(file) - prediction)
    print("\n")
    print(trueP, "True Positives")
    print(falseP, "False Positives")
    print(trueN, "True Negatives")
    print(falseN, "False Negatives")

#Test Code
print("No Obstructions", test("No_Obstructions"))
print("2 Shorts", test("2_Shorts"))
print("2 Jeans", test("2_Jeans"))
print("Shelf", test("Shelf_M"))
print("Human", test("Human_A"))


#Functions for predicting distacne, used in UI

#Getting the right model
#Switch statement better???
def regressionModel(string):
    if string == "O":
        return O
    elif string == "S2":
        return S2
    elif string == "J2":
        return J2
    elif string == "S":
        return S
    elif string == "H":
        return H
    else:
        print("Something went wrong")

#True = too close
def calculateDistance(folderName, file1, file2, model):
    RSSI1 = avgRSSI(getRSSIData(folderName, file1))
    prediction1 = RSSI1 - np.float32(regressionModel(model).intercept_)
    prediction1 /= regressionModel(model).coef_[0]
    prediction1 = pow(e, prediction1)

    RSSI2 = avgRSSIData(getRSSIData(folderName, file2))
    prediction2 = RSSI2 - np.float32(regressionModel(model).intercept_)
    prediction2 /= regressionModel(model).coef_[0]
    prediction2 = pow(e, prediction2)

    if abs(prediction2 - prediction1) <= 72.0:
        return True
    elif abs(prediction2 - prediction1) > 72.0:
        return False
    else:
        print("Something went wrong")

print(calculateDistance("No_Obstructions", "scan_0.0_77_53_20200626T150621", "scan_28.0_77_56_20200626T160113", "O"))

# Plot Graphs

#plot Linear/Logged Data
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
#ax3.scatter(df['ODistance'].astype(float),df['ORSSI'].astype(float), color = 'r')

xO = df['ODistance']
yO = df['ORSSI']
#Scatter plot via another method
#ax3.plot(xO, yO, 'o')
mO, bO = np.polyfit(xO, yO, 1)
ax3.plot(xO, mO*xO + bO, color = 'b', label = 'No Obstructions')

x2S = df['2SDistance']
y2S = df['2SRSSI']
m2S, b2S = np.polyfit(x2S, y2S, 1)
ax3.plot(x2S, m2S*x2S + b2S, color = 'r', label = '2 Shorts')

x2J = df['2JDistance']
y2J = df['2JRSSI']
m2J, b2J = np.polyfit(x2J, y2J, 1)
ax3.plot(x2J, m2J*x2J + b2J, color = 'y', label = '2 Jeans')

xS = df['SDistance']
yS = df['SRSSI']
mS, bS = np.polyfit(xS, yS, 1)
ax3.plot(xS, mS*xS + bS, color = 'm', label = 'Shelf')

xH = df['HDistance']
yH = df['HRSSI']
mH, bH = np.polyfit(xH, yH, 1)
ax3.plot(xH, mH*xH + bH, color = 'c', label = 'Human')

scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend()
ax3.set_xlabel('Distance (inches)')
ax3.set_ylabel('RSSI (decibel milliwatts)')
ax3.set_title('RSSI vs. Logged Distance for Obstructions')



#Plot Graphs pt. 2

#plot Normal/Curved Data
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
#ax4.scatter(df['realDistance'].astype(float),df['RSSI'].astype(float), color = 'g')

xO = df['ORealDistance']
yO = df['ORSSI']
#Plot scatter plot via another method
#ax3.plot(xO, yO, 'o')
ax4.plot(xO, O.coef_*np.log(xO) + O.intercept_, color = 'b', label = 'No Obstructions')

x2S = df['2SRealDistance']
y2S = df['2SRSSI']
ax4.plot(x2S, S2.coef_*np.log(x2S) + S2.intercept_, color = 'r', label = '2 Shorts')

x2J = df['2JRealDistance']
y2J = df['2JRSSI']
ax4.plot(x2J, J2.coef_*np.log(x2J) + J2.intercept_, color = 'y', label = '2 Jeans')

xS = df['SRealDistance']
yS = df['SRSSI']
ax4.plot(xS, S.coef_*np.log(xS) + S.intercept_, color = 'm', label = 'Shelf')

xH = df['HRealDistance']
yH = df['HRSSI']
ax4.plot(xH, H.coef_*np.log(xH) + H.intercept_, color = 'c', label = 'Human')

scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend()
ax4.set_xlabel('Distance (inches)')
ax4.set_ylabel('RSSI (decibel milliwatts)')
ax4.set_title('RSSI vs. Distance for Obstructions')


#Plot graphs pt. 3

#plot RSSI vs. Temperature
#Right now it is O for No_Obstructions
figure5 = plt.Figure(figsize=(5,4), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df['OTemperature'].astype(float),df['ORSSI'].astype(float), color = 'g')
scatter5 = FigureCanvasTkAgg(figure5, root)
scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax5.legend(['ORSSI'])
ax5.set_xlabel('Temperature')
ax5.set_title('Temperature Vs. RSSI')

#plot RSSI vs. Humidity
#Right now it is O for No_Obstructions
figure6 = plt.Figure(figsize=(5,4), dpi=100)
ax6 = figure6.add_subplot(111)
ax6.scatter(df['OHumidity'].astype(float),df['ORSSI'].astype(float), color = 'b')
scatter6 = FigureCanvasTkAgg(figure6, root)
scatter6.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax6.legend(['ORSSI'])
ax6.set_xlabel('Humidity')
ax6.set_title('Humidity Vs. RSSI')

#Run everything ^^^
root.mainloop()





"""
***************************************************************************************/
*    Title: From Simple to Multiple Linear Regression with Python and Scikit   source code
*    Author: Roel Peters
*    Date: 2/12/2019
*    Code version: Modified
*    Availability: https://sweetcode.io/simple-multiple-linear-regression-python-scikit/
*
***************************************************************************************/
"""

#Potential Machine Learning code for building model for my predictions
"""
df = pd.read_csv(r'data.csv')
df = df.replace('?', np.nan)
df = df.dropna()

# df = df.drop(['MAJOR','MINOR','TX POWER'], axis=1)
X = df.drop('Distance', axis=1)
y = df[['RSSI']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg = LinearRegression()
reg.fit(X_train[['RSSI']], y_train)

y_predicted = reg.predict(X_test[['RSSI']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

reg = LinearRegression()
reg.fit(X_train[['RSSI','Temp','Humid']], y_train)
y_predicted = reg.predict(X_test[['RSSI','Temp','Humid']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()
"""

#Potential exponential regression code
#Does the same thing as when I plot the curved lines above (ln. 337)
"""
def func(x, a, b, c):
    value = a * np.exp(-b * x) + c
    return value

xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
popt = [ 2.55423706,  1.35190947,  0.47450618]
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
popt = [ 2.43708906,  1.        ,  0.35015434]
plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""
