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

#exponential
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
#Basic linear
"""
x = np.log(getAllDistances())
y = getAllRSSI("Short_A")
plt.plot(x, y, 'o')

m, b = np.polyfit(x, y, 1)

plt.plot(x, m*x + b)
plt.show()
"""

#My multilinear

Bluetooth = {'RSSI': getAllRSSI("Short_A"),
                'Distance': np.log(getAllDistances("short_A")),
                'realDistance': getAllDistances("short_A"),
                'Temperature': np.log(getAllTemp("Short_A")),
                'Humidity': np.log(getAllHumidity("Short_A")),
                }

#df = pd.DataFrame(Bluetooth,columns=['RSSI','Distance','Temperature','Humidity'])
df = pd.DataFrame(Bluetooth,columns=['RSSI', 'Distance', 'realDistance'])
print (df)

#X = df[['Distance','Temperature', 'Humidity']]
X = df[['Distance']]
Y = df['RSSI']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# Print header stuff
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)

# New RSSI label and input box
label1 = tk.Label(root, text='Type RSSI: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

"""
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
"""
"""
# Calculate prediction w/ Tempertaure & Humidity
def values():
    global RSSI #our 1st input variable
    RSSI = np.float32(entry1.get())

    global Temperature #our 2nd input variable
    Temperature = np.float32(entry2.get())
    Temperature = np.log(Temperature)

    global Humidity #our 3rd input variable
    Humidity = np.float32(entry3.get())
    Humidity = np.log(Humidity)


    prediction = RSSI - np.float32(regr.intercept_)
    prediction -= (regr.coef_[1] * Temperature)
    prediction -= (regr.coef_[2] * Humidity)
    prediction /= regr.coef_[0]
    prediction = pow(e, prediction)
    Prediction_result  = ('Predicted Distance: ', prediction)
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
"""
def values():
    global RSSI #our 1st input variable
    RSSI = np.float32(entry1.get())

    prediction = RSSI - np.float32(regr.intercept_)
    prediction /= regr.coef_[0]
    prediction = pow(e, prediction)

    prediction2 = RSSI + 30.0
    prediction2 /= -10
    prediction2 = pow(e, prediction2)

    Prediction_result  = ('Predicted Distance: ', prediction, prediction2)
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
"""
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
        prediction = RSSI - np.float32(regr.intercept_)
        #prediction -= (regr.coef_[1] * Temperature)
        #prediction -= (regr.coef_[2] * Humidity)
        prediction /= regr.coef_[0]
        prediction = pow(e, prediction)
        if truePositive(actual, prediction):
            trueP += 1
        if falsePositive(actual, prediction):
            falseP += 1
        if trueNegative(actual, prediction):
            trueN += 1
        if falseNegative(actual, prediction):
            falseN += 1
        #print(getDistance(file) - prediction)
    print("\n")
    print(trueP, "True Positives")
    print(falseP, "False Positives")
    print(trueN, "True Negatives")
    print(falseN, "False Negatives")
test("Short_A")
"""
"""
def test0(folderName):
    for file in getAllFolderFiles(folderName)
        Temperature = np.log(getTemp(file))
        Humidity = np.log(getHumidity(file))
        value = np.float32(regr.intercept_)
        value += regr.coef_[1] * Temperature
        value += regr.coef_[0] * Humidity
        value

test0()
"""

button1 = tk.Button (root, text='Predict Distance',command=values, bg='orange') # button to call the 'values' command above
canvas1.create_window(270, 170, window=button1)

# Plot Graphs

#plot 1st scatter
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['Distance'].astype(float),df['RSSI'].astype(float), color = 'r')

x = np.log(getAllDistances("Short_A"))
y = getAllRSSI("Short_A")
#ax3.plot(x, y, 'o')
m, b = np.polyfit(x, y, 1)
ax3.plot(x, m*x + b)
ax3.plot(x, -11*x + -30)
#plt.show()

scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend(['RSSI'])
ax3.set_xlabel('Distance')
ax3.set_title('Distance Vs. RSSI')




#plot 2nd scatter
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(getAllDistances("Short_A").astype(float),getAll.astype(float), color = 'g')

x = getAllDistances("Short_A")
y = getAllRSSI("Short_A")
#ax3.plot(x, y, 'o')

ax4.plot(x, regr.coef_[0]*np.log(x) + regr.intercept_)
#plt.show()

scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['RSSI'])
ax4.set_xlabel('Distance')
ax4.set_title('Distance Vs. RSSI')

"""
#plot 2nd scatter
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['Temperature'].astype(float),df['RSSI'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['RSSI'])
ax4.set_xlabel('Temperature')
ax4.set_title('Temperature Vs. RSSI')

#plot 3rd scatter
figure5 = plt.Figure(figsize=(5,4), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df['Humidity'].astype(float),df['RSSI'].astype(float), color = 'b')
scatter5 = FigureCanvasTkAgg(figure5, root)
scatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax5.legend(['RSSI'])
ax5.set_xlabel('Humidity')
ax5.set_title('Humidity Vs. RSSI')
"""

root.mainloop()


#Machine Learning
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
