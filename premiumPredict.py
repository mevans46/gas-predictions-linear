import numpy as np
import pandas as pd
from tkinter import *

class MachineLearn:
    def init(self) -> None:
        pass

    #predict function
    def predict(self,X, w):
        return X * w

    #calculate loss
    def loss(self,X, Y, w):
        return np.average((self.predict(X,w) - Y) ** 2)

    #train function
    def train(self,X, Y, iterations, lr):
        w = 0
        for i in range(iterations):
            current_loss = self.loss(X, Y, w)
            print("Iteration {} => Loss: {}".format(i, current_loss))
            lossLabel.set(current_loss)
            if self.loss(X, Y, w + lr) < current_loss:
                w += lr
            elif self.loss(X, Y, w - lr) < current_loss:
                w -= lr
            else:
                weightLabel.set(w)
                print("Data provided by Kaggle")
                return w

        raise Exception("Couldn't converge within {} iterations".format(iterations))

    def change_price_label(self,X,w):
        pGasPrice.set(str(ML.predict(float(X), w)))






# Import the dataset
df = pd.read_csv("data/GAS_DATA.csv")
X = df["R1"]
Y = df["P1"]
ML = MachineLearn()

# Train the system
#w = ML.train(X,Y,iterations=10000, lr=0.01)
#print("\nw={}".format(w))

# Predict the price of gas
#p = input("Enter Current Regular Gas Price: ")
#print("Predicted Premium Gas Price: {}".format(ML.predict(float(p), w)))




#create window
root = Tk()

root.geometry("350x175")
root.title("Premium Gas Prices")
text = Text(root)

#define text
pGasPrice = StringVar()
premiumLabel = StringVar()
regularLabel = StringVar()
weightLabel = StringVar()
weightLabelText = StringVar()
lossLabel = StringVar()
lossLabelText = StringVar()

#set text
weightLabelText.set("Weight: ")
premiumLabel.set("Predicted Premium Gas Price: ")
regularLabel.set("Please enter price of regular gas: ")
lossLabelText.set("Loss: ")


#define widgets
labelLT = Label( root, textvariable=lossLabelText, relief=RAISED,bg='#fff', fg='#f00' )
labelL = Label( root, textvariable=lossLabel, relief=RAISED,bg='#fff', fg='#f00' )
labelWt = Label( root, textvariable=weightLabelText, relief=RAISED,bg='#fff', fg='#f00' )
labelW = Label( root, textvariable=weightLabel, relief=RAISED,bg='#fff', fg='#f00' )
labelR = Label( root, textvariable=regularLabel, relief=RAISED,bg='#fff', fg='#f00' )
entry = Entry(root)
B = Button(root, text ="Press to Train",bg='#fff', fg='#f00' , command = lambda : ML.change_price_label(entry.get(),ML.train(X,Y,iterations=1000,lr = 0.01)))
labelP = Label( root, textvariable=premiumLabel, relief=RAISED,bg='#fff', fg='#f00' )
labelAnswer = Label( root, textvariable=pGasPrice, relief=RAISED,bg='#fff', fg='#f00' )

#pack/ add widgets to screen
labelR.grid(column=0,row=0, pady= 3,padx=5)
entry.grid(column=1,row=0, pady=3)
B.grid(column=0,row=2, pady=10)
labelLT.grid(column=0,row=3, pady=3)
labelL.grid(column=1,row=3, pady=3)
labelWt.grid(column=0,row=4, pady=3)
labelW.grid(column=1,row=4, pady=3)
labelP.grid(column=0,row=5, pady=3)
labelAnswer.grid(column=1,row=5, pady=10)

#loop window
root.mainloop()