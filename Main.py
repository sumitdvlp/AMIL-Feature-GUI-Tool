__author__ = 'Lisa'

import tkinter
from tkinter import *
import tkinter.filedialog
from tkinter.tix import *
import csv
import USFS
import Go
import os
class Application(Frame):
    csvResponse = 0
    csvRealArt = 0
    datazip = 0
    toPrint = " "
    w = 0
    initialTime = 1

    def USFSClick(self, csvResponse, csvRealArt, datazip):
        print(csvResponse)
        print(csvRealArt)
        Application.printToWindow(self, "Start running USFS")
        if csvResponse != 0 and csvRealArt != 0 and datazip != 0:
            myUSFS = USFS.USFS()
            self.printToWindow("Start running USFS")
            self.printToWindow(myUSFS.printla())
            #myUSFS.run(csvResponse, csvRealArt, datazip)
            print("Finished running USFS")
        else:
            print("Missing response and/or realart CSV")
    def GoClick(self, csvResponse, csvRealArt, datazip):
        Application.printToWindow(self, "Starting Application")
        if csvResponse != 0 and csvRealArt != 0 and datazip != 0:
            myGo = Go.Go()
            processedFiles = myGo.processFiles(csvResponse, csvRealArt, datazip)
            varKept = 0
            if self.PCVar.get() == 1:
                self.printToWindow("Dimensionality Reduction Enabled")
                self.printToWindow("Beginning Dimensionality Reduction")
                varKept = myGo.DimReduction(float(self.PCVarToKeep.get()), processedFiles["response"],
                                            processedFiles["dataList"])
                self.printToWindow(varKept)
            else:
                self.printToWindow("Dimensionality Reduction Disabled")

            #get classifiers
            classifiers = []
            if self.LDAVar.get() == 1:
                classifiers.append("lda")
            if self.QDAVar.get() == 1:
                classifiers.append("qda")
            if self.SVMVar.get() == 1:
                classifiers.append("svm")

            #feature selection
            fs = 0
            if self.FSVar.get() == "USFS":
                if self.PCVar.get() == 1:
                    fs = myGo.runUSFS(processedFiles["response"], processedFiles["realArt"], varKept["scoreTotal"],
                                 classifiers)
                else:
                    fs = myGo.runUSFS(processedFiles["response"], processedFiles["realArt"], processedFiles["data"],
                                 classifiers)
            elif self.FSVar.get() == "LASSO":
                if self.PCVar.get() == 1:
                    fs = myGo.runLASSO(varKept["scoreTotal"], processedFiles["response"])
                else:
                    print("running with data")
                    fs = myGo.runLASSO(processedFiles["data"], processedFiles["response"])
            elif self.FSVar.get() == "Elastic Net":
                if self.PCVar.get() == 1:
                    fs = myGo.runElasticNet(varKept["scoreTotal"], processedFiles["response"])
                else:
                    fs = myGo.runElasticNet(processedFiles["data"], processedFiles["response"])
            print('fs is',fs)
            self.FSOutput(fs["bestFeatureSummary"], classifiers)

            #myUSFS.run(csvResponse, csvRealArt, datazip)
            self.printToWindow("Application Complete")

    def FSOutput(self, bestFeatureSummary, classifier):
        for i in range(0, len(classifier)):
            title = "SummarybestFeatures_" + classifier[i]
            accuracyString = "Accuracy: " + str(bestFeatureSummary[i][0][0])
            accLowClassString = "Accuracy (Low Class): " + str(bestFeatureSummary[i][1][0])
            accHighString = "Accuracy (High Class): " + str(bestFeatureSummary[i][2][0])
            self.printToWindow(title)
            self.printToWindow(accuracyString)
            self.printToWindow(accLowClassString)
            self.printToWindow(accHighString)

            bfsShape = bestFeatureSummary[i].shape
            for j in range(3, bfsShape[0]):
                featureString = "Feature at indice " + str(bestFeatureSummary[i][j][0]) + " contributed " + \
                                str(bestFeatureSummary[i][j][1]) + " percentage to accuracy"
                self.printToWindow(featureString)

    def printToWindow(self, string):
        if Application.initialTime == 1:
            frame = Frame(root, width=975, height=700)
            root.configure(background='black')
            frame.configure(background='black')
            frame.pack()
            self.swin = ScrolledWindow(frame, width=975, height=700)
            self.swin.configure(background='black')
            self.swin.pack()
            self.win = self.swin.window
            self.w = Message(self.win, text=string, width=400)
            self.w.pack()
            Application.initialTime = 0
        else:
            self.w = Message(self.win, text=string, width=400)
            self.w.pack()

    def createWidgets(self):

        frameActionButtons = Frame(root)

        inputActionButtons = Frame(frameActionButtons)
        self.InputLabel = Label(inputActionButtons, text="Input")

        self.CSVResponseButton = Button(inputActionButtons)
        self.CSVResponseButton["text"] = "Upload Response CSV"
        self.CSVResponseButton["command"] = self.openfileResponse

        self.CSVRealArtButton = Button(inputActionButtons)
        self.CSVRealArtButton['text'] = "Upload Real Art CSV"
        self.CSVRealArtButton["command"] = self.openfileRealArt

        self.DataUploadButton = Button(inputActionButtons)
        self.DataUploadButton["text"] = "Upload Data as zip file"
        self.DataUploadButton["command"] = self.openfileData

        frameActionButtons.pack(side="left", fill=BOTH)

        inputActionButtons.pack(fill=X)
        self.InputLabel.pack(fill=X)
        self.CSVResponseButton.pack(side=LEFT)
        self.CSVRealArtButton.pack(side=LEFT)
        self.DataUploadButton.pack(side=LEFT)
        #self.USFSButton.pack(fill=X)

        dimReductionFrame = Frame(frameActionButtons)
        self.dimRedLabel = Label(dimReductionFrame, text="Dimensionality Reduction")
        self.PCLabel = Label(dimReductionFrame, text="Apply PCA: ")
        self.PCVar = IntVar()
        self.PCButton = Checkbutton(dimReductionFrame, variable=self.PCVar)
        self.PCVarToKeep = StringVar()
        self.variationEntry = Entry(dimReductionFrame, textvariable=self.PCVarToKeep)
        self.variationLabel = Label(dimReductionFrame, text="% variation to keep")

        dimReductionFrame.pack(fill=X)
        self.dimRedLabel.pack(fill=X)
        self.PCLabel.pack(side=LEFT)
        self.PCButton.pack(side=LEFT)
        self.variationEntry.pack(side=LEFT)
        self.variationLabel.pack(side=LEFT)

        featureSelectionFrame = Frame(frameActionButtons)
        self.FSLabel = Label(featureSelectionFrame, text="Feature Selection")
        self.boxvar = StringVar()

        #var.set(FSOptions[0])
        #FSOptionMenu = OptionMenu(featureSelectionFrame, var, tuple(FSOptions))
        self.FSVar = StringVar()
        self.FSOptionMenu = ComboBox(featureSelectionFrame, variable=self.FSVar)
        self.FSOptionMenu.insert(END, 'USFS')
        self.FSOptionMenu.insert(END, 'LASSO')
        self.FSOptionMenu.insert(END, 'Elastic Net')
        self.FSOptionMenu.insert(END, 'PSO')
        self.FSOptionMenu.insert(END, 'Simulated Annealing')
        #self.FSOptionMenu['entry']=('USFS', 'LASSO', 'Elastic Net', 'PSO', 'Simulated Annealing')


        featureSelectionFrame.pack(fill=X)
        self.FSLabel.pack(fill=X)
        self.FSOptionMenu.pack()

        classificationAlgFrame = Frame(frameActionButtons)
        self.CALabel = Label(classificationAlgFrame, text="Classification Algorithms")
        self.CAListBox = Menubutton(classificationAlgFrame, text="Select Here", relief=RAISED)

        self.CAListBox.menu = Menu(self.CAListBox, tearoff=0)
        self.CAListBox["menu"] = self.CAListBox.menu

        self.LDAVar = IntVar()
        self.QDAVar = IntVar()
        self.SVMVar = IntVar()

        self.CAListBox.menu.add_checkbutton(label="LDA", variable=self.LDAVar)
        self.CAListBox.menu.add_checkbutton(label="QDA", variable=self.QDAVar)
        self.CAListBox.menu.add_checkbutton(label="SVM", variable=self.SVMVar)

        classificationAlgFrame.pack(fill=X)
        self.CALabel.pack(fill=X)
        self.CAListBox.pack()

        GoFrame = Frame(frameActionButtons)

        self.GoButton = Button(GoFrame, text="GO", fg="red")
        #self.GoButton["text"] = "GO"
        self.GoButton["fg"] = "red"
        #self.USFSButton["command"] = lambda: self.USFSClick(csvResponse=Application.csvResponse, csvRealArt=Application.csvRealArt, datazip=Application.datazip)
        self.GoButton["command"] = lambda: self.GoClick(csvResponse=Application.csvResponse, csvRealArt=Application.csvRealArt, datazip=Application.datazip)
        GoFrame.pack(fill=X)
        self.GoButton.pack(side=BOTTOM)


        self.LDALabel = Label(self, text="Summary Best LDA's")
        #self.LDALabel.grid(row=4, sticky=W)


        #frame = Frame(root, width=500, height=600)
        #root.configure(background='black')
        #frame.configure(background='black')
        #frame.pack()
        #swin = ScrolledWindow(frame, width=500, height=600)
        #swin.configure(background='black')
        #swin.pack()
        #win = swin.window
        #Application.w = Message(win, text=" ".join(Application.toPrint), width=500)
        #Application.w.pack()


        Application.printToWindow(self, "Helloworld")


        ##################
        #top = self.winfo_toplevel()
        #self.menuBar = Menu(top)
        #top["menu"] = self.menuBar
        #self.subMenu = Menu(self.menuBar)
        #self.subMenu.add_command(label="Open", command=self.openfile)
        #self.subMenu.add_separator()
        #self.subMenu.add_command( label = "Read Data",command = self.readCSV)
        #self.menuBar.add_cascade(label = "File", menu = self.subMenu)

    def readCSV(self):
        self.filename = tkinter.filedialog.askopenfilename()
        f = open(self.filename,"rb")
        read = csv.reader(f, delimiter = ",")
        buttons = read.next()
        print
        for btn in buttons:
            new_btn = Button(self, text=btn, command=self.btnClick)
            new_btn.pack()

    def btnClick(self):
        pass

    def openfileResponse(self):
        print(hasattr(tkinter, "filedialog"))
        filename = tkinter.filedialog.askopenfilename(parent=root)
        csvFilePath = tkinter.Label(root, text=filename)
        #csvFilePath.grid()
        Application.csvResponse = filename

    def openfileRealArt(self):
        filename = tkinter.filedialog.askopenfilename(parent=root)
        csvFilePath = tkinter.Label(root, text=filename)
        #csvFilePath.grid()
        Application.csvRealArt = filename

    def openfileData(self):
        filename = tkinter.filedialog.askopenfilename(parent=root)
        csvFilePath = tkinter.Label(root, text=filename)
        #csvFilePath.grid()
        Application.datazip = filename
        print('==Sumit==',filename)


    def __init__(self, master=None):
        master.minsize(width=1350, height=625)
        Frame.__init__(self, master)

        self.createWidgets() ## this function
        master.columnconfigure(0, weight=1)
        #self.grid(sticky=(N, S, E, W))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()