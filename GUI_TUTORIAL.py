import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import Main2
import tkinter as tk
from tkinter import ttk
from tkinter import StringVar,OptionMenu,Label
from scipy import stats

LARGE_FONT = ("Verdana", 12)


class SeaofBTCapp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Feature Selection GUI")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        #for F in (StartPage, FSCORE, CMIM, JMI):
        for F in (StartPage, FSCORE):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Select Feature Selection Algorithm", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = ttk.Button(self, text="Fisher Score",
                            command=lambda: controller.show_frame(FSCORE))
        button.pack()

        button2 = ttk.Button(self, text="Conditional Mutual Information Maximization",
                             command=lambda: controller.show_frame(CMIM))
        button2.pack()

        button3 = ttk.Button(self, text="Joint Mutual Information",
                             command=lambda: controller.show_frame(JMI))
        button3.pack()

        button4 = ttk.Button(self, text="Exit !!",
                            command=quit)
        button4.pack()



        myG = Main2.FeatureSelect()
        idx = myG.F_SCORE()
        Fs10, Fs20,Fs11,Fs21 = myG.get_scaled_values(idx)

#        idx=myG.CMIM()
        Cm10, Cm20,Cm11,Cm21 = myG.get_scaled_values(idx)

#        idx = myG.JMI()
        Jm10, Jm20, Jm11, Jm21 = myG.get_scaled_values(idx)

        print('Fs10==',Fs10,'Fs20==',Fs20,'Fs11--',Fs11,'Fs21--',Fs21)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))

        axes[0][0].plot(Fs10, Fs20, 'ro')
        axes[0][0].plot(Fs11, Fs21, 'bs')
        axes[0][0].axis([0, 1, 0, 1])
        axes[0][0].set_title('Fisher Score')

        axes[0][1].plot(Cm10, Cm20, 'ro')
        axes[0][1].plot(Cm11, Cm21, 'bs')
        axes[0][1].axis([0, 1, 0, 1])
        axes[0][1].set_title('CMIM Score')

        axes[1][0].plot(Jm10, Jm20, 'ro')
        axes[1][0].plot(Jm11, Jm21, 'bs')
        axes[1][0].axis([0, 1, 0, 1])
        axes[1][0].set_title('JMI Score')

#        plt.setp(axes, xticks=[y + 1 for y in range(len(Fe1))], xticklabels=['Y=0', 'y=1'])

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class FSCORE(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        myG = Main2.FeatureSelect()
        idx = myG.F_SCORE()
        ttest=1
        Fe1, Fe2 = myG.get_values(idx)
        txttest='Fisher Score , T test value is '+ascii(ttest)
        label = tk.Label(self, text=txttest, font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button4 = ttk.Button(self, text="Exit !!",command=quit)
        button4.pack()

        myG = Main2.FeatureSelect()
        idx = myG.F_SCORE()
        ttest = 1
        Fe1, Fe2 = myG.get_values(idx)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))
        axes[0][0].violinplot(Fe1, showmeans=False, showmedians=True)
        axes[0][0].set_title('Feature 1')
        axes[0][1].violinplot(Fe2, showmeans=False, showmedians=True)
        axes[0][1].set_title('Feature 2')
        axes[1][0].boxplot(Fe1)
        axes[1][0].set_title('Feature 1')
        axes[1][1].boxplot(Fe2)
        axes[1][1].set_title('Feature 2')

        plt.setp(axes, xticks=[y + 1 for y in range(len(Fe1))], xticklabels=['Y=0', 'y=1'])
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class CMIM(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        myG=Main2.FeatureSelect()
        idx,ttest=myG.CMIM()
        txttest='Conditional Mutual Information Maximization, T test value is '+ascii(ttest)

        label = tk.Label(self, text=txttest, font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button4 = ttk.Button(self, text="Exit !!",
                            command=quit)
        button4.pack()

        myG=Main2.FeatureSelect()
        idx,ttest=myG.CMIM()
        Fe1,Fe2=myG.get_values(idx)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))
        axes[0][0].violinplot(Fe1, showmeans=False, showmedians=True)
        axes[0][0].set_title('Feature 1')

        axes[0][1].violinplot(Fe2, showmeans=False, showmedians=True)
        axes[0][1].set_title('Feature 2')
        axes[1][0].boxplot(Fe1)
        axes[1][0].set_title('Feature 1')
        axes[1][1].boxplot(Fe2)
        axes[1][1].set_title('Feature 2')


        plt.setp(axes, xticks=[y + 1 for y in range(len(Fe1))], xticklabels=['Y=0', 'y=1'])

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class JMI(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        myG=Main2.FeatureSelect()
        idx,ttest=myG.JMI()
        txttest='Joint Mutual Information, T test value is '+ascii(ttest)

        label = tk.Label(self, text=txttest, font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button4 = ttk.Button(self, text="Exit !!",
                            command=quit)
        button4.pack()

        myG=Main2.FeatureSelect()
        idx,ttest=myG.JMI()
        Fe1,Fe2=myG.get_values(idx)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))
        axes[0][0].violinplot(Fe1, showmeans=False, showmedians=True)
        axes[0][0].set_title('Feature 1')

        axes[0][1].violinplot(Fe2, showmeans=False, showmedians=True)
        axes[0][1].set_title('Feature 2')
        axes[1][0].boxplot(Fe1)
        axes[1][0].set_title('Feature 1')
        axes[1][1].boxplot(Fe2)
        axes[1][1].set_title('Feature 2')


        plt.setp(axes, xticks=[y + 1 for y in range(len(Fe1))], xticklabels=['Y=0', 'y=1'])

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# myG=Main2.FeatureSelect()
# idx=myG.JMI()
# myG.get_values(idx)
# myG.draw()
app = SeaofBTCapp()
app.mainloop()