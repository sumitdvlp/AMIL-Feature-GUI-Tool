import tkinter
import tkinter.constants as Tkconstants
from tkinter import ttk,Frame
from tkinter import StringVar,IntVar,Checkbutton,CHECKBUTTON
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# Feature Selection method
import Main2

FilterOp = {'Mutual-Information', 'Entropy'}
WrapperOp = {'Sequntial-Forward-Selection', 'Particale-Swarm-Optimization'}
EmbeddedOp = {'Lasso', 'ElasticNet'}

class Adder(ttk.Frame):
    XPath = ''
    yPath = ''
    dimRedValue=''
    featSelecValue=''
    featSelMethValue=''
    ClassFiersTest=[]
    histGram=0
    scatPlot=0
    impFeat=0
    AccMod=0
    missClasInst=0
    histN=0

    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def on_quit(self):
        """Exits program."""
        quit()

    def init_gui(self):
        """Builds GUI."""
        self.root.title('AMIL - Feature Selection Tool')
        self.root.option_add('*tearOff', 'FALSE')
        self.root.geometry('800x1000+600+600')

        self.grid(column=0, row=0, sticky='nsew')

        self.menubar = tkinter.Menu(self.root)

        self.menu_file = tkinter.Menu(self.menubar)
        self.menu_file.add_command(label='Exit', command=self.on_quit)
        self.menu_file.add_command(label='Open', command=self.OpenFile)


        self.menu_edit = tkinter.Menu(self.menubar)

        self.menubar.add_cascade(menu=self.menu_file, label='File')
        self.menubar.add_cascade(menu=self.menu_edit, label='Edit')
        self.menubar.add_cascade(menu=self.menu_edit, label='Exit',command=self.on_quit)


        self.root.config(menu=self.menubar)


        ttk.Label(self, text='Data Input:').grid(column=0, row=0,columnspan=4)

        ttk.Separator(self, orient='horizontal').grid(column=0,row=1, columnspan=4, sticky='ew')

        ttk.Label(self, text='Dependent Variables(X)').grid(column=0, row=2,sticky='w')
        self.file_DiaY = ttk.Button(self, text="Choose X File",command=lambda: self.OpenFile(1))
        self.file_DiaY.grid(column=1,row=2)

        ttk.Label(self, text='Independent Variables(y)').grid(column=0, row=3,sticky='w')
        self.file_DiaX = ttk.Button(self, text="Choose Y File",command=lambda: self.OpenFile(2))
        self.file_DiaX.grid(column=1,row=3)

        ttk.Separator(self, orient='horizontal').grid(column=0,row=4, columnspan=4, sticky='ew')

        ttk.Label(self, text='Dimension Reduction?').grid(column=0, row=5,sticky='w')
        tkvarDR=StringVar()
        dimenReducCh = {'No', 'Yes, PCA', 'Yes, ICA'}
        tkvarDR.set("No")
        self.dimenReduc=ttk.OptionMenu(self,tkvarDR,*dimenReducCh,command=self.dimRedVal)
        self.dimenReduc.grid(column=1, row=5)

        ttk.Separator(self, orient='horizontal').grid(column=0,row=6, columnspan=4, sticky='ew')

        ttk.Label(self, text='Feature Selection Type?').grid(column=0, row=7,sticky='w')
        tkvarFS=StringVar()
        FeatSeleCh = {'Filter', 'Wrapper', 'Embedded'}
        tkvarFS.set('Wrapper')
        ttk.OptionMenu(self,StringVar(),*FeatSeleCh,command=self.featSelecVal).grid(column=1, row=7)
        tkvarFS.get()
        ttk.Label(self, text='Feature Selection Method?').grid(column=4, row=7, sticky='w')

        ttk.Separator(self, orient='horizontal').grid(column=0,row=9, columnspan=4, sticky='ew')

        ttk.Label(self, text='Output :').grid(column=0, row=10, sticky='w')

        self.histGram = IntVar()
        ttk.Checkbutton(self, text="Histogram of Top n Features", variable=self.histGram).grid(row=12, sticky='w')
        self.histGram.trace_variable("w",self.HistGramFxn)
        ttk.Label(self, text='| n = ').grid(column=3, row=12, sticky='w')
        self.histN=ttk.Entry(self).grid(column=4,row=12,pady=2,sticky='w')

        self.scatPlot = IntVar()
        ttk.Checkbutton(self, text="Scatler Plot of Top n Features", variable=IntVar()).grid(row=13, sticky='w')
        self.scatPlot.trace_variable("w",self.ScatPlotFxn)

        ttk.Label(self, text='| Select Plot Type = ').grid(column=3, row=13, sticky='w')
        tkvarPT=StringVar()
        ttk.OptionMenu(self,tkvarPT,'Make 2D PLot', 'Make 3D PLot').grid(column=4,row=13,sticky='w')

        self.impFeat=IntVar()
        ttk.Checkbutton(self, text="Show Importance of Each Feature", variable=self.impFeat).grid(row=14, sticky='w')
        self.impFeat.trace_variable("w",self.ImpFeatFxn)

        self.AccMod=IntVar()
        ttk.Checkbutton(self, text="Show Accuracy of Model", variable=self.AccMod).grid(row=15, sticky='w')
        self.AccMod.trace_variable("w",self.AccModFxn)

        self.MisInt=IntVar()
        ttk.Checkbutton(self, text="Misclassfied Instances", variable=self.MisInt).grid(row=16, sticky='w')
        self.MisInt.trace_variable("w",self.MisIntFxn)

        ttk.Button(self, text="Submit",command=lambda: self.collectValues()).grid(row=17,column=0,columnspan=16)
        ttk.Button(self, text="Quit",command=lambda: self.on_quit()).grid(row=17,column=5,columnspan=2)

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def HistGramFxn(self,*args):
        self.histGram=self.histGram.get()
    def ScatPlotFxn(self,*args):
        self.scatPlot=self.scatPlot.get()
    def MisIntFxn(self,*args):
        self.MisInt=self.MisInt.get()
    def AccModFxn(self,*args):
        self.AccMod=self.AccMod.get()
    def ImpFeatFxn(self,*args):
        self.impFeat=self.impFeat.get()
    def collectValues(self):

        frameScatPlot = Frame(self)
        frameScatPlot.grid(column=0, row=25, columnspan=100, sticky=Tkconstants.NSEW)
        frameScatPlot.rowconfigure(50, weight=100)
        frameScatPlot.columnconfigure(1, weight=1)

        frameFeatHist = Frame(self)
        frameFeatHist.grid(column=0, row=19, columnspan=100, sticky=Tkconstants.NSEW)
        frameFeatHist.rowconfigure(50, weight=100)
        frameFeatHist.columnconfigure(1, weight=1)

        n=8
        myG = Main2.FeatureSelect()
        idx = myG.F_SCORE()
        featureList=myG.get_Feature_List(idx,n)
        self.draw_Feat_Hist(frameFeatHist,featureList,n)
        self.draw_Scatt_Plot(frameScatPlot,featureList,n)

    def draw_Scatt_Plot(self,frameScatPlot,featureList,n):
        fig, axes = plt.subplots(1, n-1, sharex=True, figsize=((n-1)*4, 3), squeeze=False)

        for i in range(0,n-1):

            F10=MinMaxScaler().fit_transform(featureList[i][0][0])
            F11 =MinMaxScaler().fit_transform(featureList[i][0][1])
            F20=MinMaxScaler().fit_transform(featureList[i+1][0][0])
            F21 =MinMaxScaler().fit_transform(featureList[i+1][0][1])
            print('F10-',F10,'F20-', F20)
            axes[0][i].plot(F10, F20,'go',label='beningn')
            axes[0][i].plot(F11, F21,'r^',label='malign')
            axes[0][i].legend(loc='best')

            axes[0][i].axis([0, 1, 0, 1])
            prn='Feature'+str(i+1)+' VS '+'Feature'+str(i+2)
            axes[0][i].set_title(prn)

        fig.tight_layout()
        self.addScrollingFigure(fig, frameScatPlot)
        self.changeSize(fig,0.8)

    def draw_Feat_Hist(self,frame,featureList,n):

        fig, ax = plt.subplots(1, n, sharex=True, figsize=(n*4, 3), squeeze=False)
        
        for i in range(0,n):
            feat=featureList[i]
            ax[0][i-1].violinplot(feat[0], showmeans=False, showmedians=True)
            pri=str(i)+' p-value - ' +str(feat[1])
            ax[0][i-1].set_title(pri)
        fig.tight_layout()
        self.addScrollingFigure(fig, frame)
        self.changeSize(fig,0.8)

    def changeSize(self,figure, factor):
        global canvas, mplCanvas, interior, interior_id, frame, cwid
        oldSize = figure.get_size_inches()
        figure.set_size_inches([factor * s for s in oldSize])
        wi, hi = [i * figure.dpi for i in figure.get_size_inches()]

        mplCanvas.config(width=wi, height=hi)
        canvas.itemconfigure(cwid, width=wi, height=hi)
        canvas.config(scrollregion=canvas.bbox(Tkconstants.ALL), width=200, height=200)
        figure.canvas.draw()

    def addScrollingFigure(self,figure, frame):
        global canvas, mplCanvas, interior, interior_id, cwid
        # set up a canvas with scrollbars
        canvas = tkinter.Canvas(frame)
        canvas.grid(row=1, column=1, sticky=Tkconstants.NSEW)

        xScrollbar = tkinter.Scrollbar(frame, orient=Tkconstants.HORIZONTAL)
        yScrollbar = tkinter.Scrollbar(frame)

        xScrollbar.grid(row=2, column=1, sticky=Tkconstants.EW)
        yScrollbar.grid(row=1, column=2, sticky=Tkconstants.NS)

        canvas.config(xscrollcommand=xScrollbar.set)
        xScrollbar.config(command=canvas.xview)
        canvas.config(yscrollcommand=yScrollbar.set)
        yScrollbar.config(command=canvas.yview)

        figAgg = FigureCanvasTkAgg(figure, canvas)
        mplCanvas = figAgg.get_tk_widget()

        cwid = canvas.create_window(0, 0, window=mplCanvas, anchor=Tkconstants.NW)
        #changeSize(figure, 1)

    def histGramChk(self,value):
        print(self.featSelecValue)
    def dimRedVal(self,value):
        self.dimRedValue=value
    def featSelecVal(self,value):
        self.featSelecValue=value
        tkvar = StringVar()
        tkvar.set('')
        if(self.featSelecValue=='Filter'):
            FeatSelMethCh=FilterOp
        elif(self.featSelecValue=='Wrapper'):
            FeatSelMethCh=WrapperOp
        elif(self.featSelecValue=='Embedded'):
            FeatSelMethCh=EmbeddedOp
        self.FeatSelMethOp = ttk.OptionMenu(self, tkvar, *FeatSelMethCh, command=self.featSelMethVal)
        self.FeatSelMethOp.grid(column=6, row=7)

        if(self.featSelecValue in('Embedded','Wrapper')):
            ttk.Label(self, text='Classifier to Test:').grid(column=7, row=7, sticky='w')
            ttk.Checkbutton(self, text="Linear Discriment Analysis (LDA)", variable=IntVar).grid(row=8,column=7, sticky='w')
            ttk.Checkbutton(self, text="Quadratic Discriment Analysis (QDA)", variable=IntVar).grid(row=9,column=7, sticky='w')
            ttk.Checkbutton(self, text="Support Vector Machine (SVM)", variable=IntVar).grid(row=10,column=7, sticky='w')

    def featSelMethVal(self,value):
        self.featSelMethValue=value
        print(self.featSelMethValue)
    def OpenFile(self,x):
        name = filedialog.askopenfilename(initialdir="/home/launch/Desktop/Share/",
                               filetypes =(("Csv File", "*.csv"),("All Files","*.*")),
                               title = "Choose a file."
                               )
        print (name)
        #Using try in case user types in unknown file or closes without choosing a file.
        try:
            with open(name,'r') as UseFile:
                print(UseFile.read())
        except:
            print("No file exists")
        if x==1:
            self.XPath=name
        else:
            self.yPath=name


class MyCheckButton(ttk.Checkbutton):
    def __init__(self,*args,**kwargs):
        self.var=kwargs.get('variable',IntVar())
        kwargs['variable']=self.var
        Checkbutton.__init__(self,*args,**kwargs)

    def is_checked(self):
        return self.var.get()

if __name__ == '__main__':
    root = tkinter.Tk()
    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1)
    Adder(root)
    root.mainloop()
