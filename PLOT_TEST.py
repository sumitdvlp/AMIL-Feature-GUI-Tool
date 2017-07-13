import math
import sys
if sys.version_info[0] < 3:
  from Tkinter import Tk, Button, Frame, Canvas, Scrollbar
  import Tkconstants
else:
  from tkinter import Tk, Button, Frame, Canvas, Scrollbar
  import tkinter.constants as Tkconstants

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pprint, inspect

def addScrollingFigure(figure, frame):
  global canvas, mplCanvas, interior, interior_id, cwid
  # set up a canvas with scrollbars
  canvas = Canvas(frame)
  canvas.grid(row=1, column=1, sticky=Tkconstants.NSEW)

  xScrollbar = Scrollbar(frame, orient=Tkconstants.HORIZONTAL)
  yScrollbar = Scrollbar(frame)

  xScrollbar.grid(row=2, column=1, sticky=Tkconstants.EW)
  yScrollbar.grid(row=1, column=2, sticky=Tkconstants.NS)

  canvas.config(xscrollcommand=xScrollbar.set)
  xScrollbar.config(command=canvas.xview)
  canvas.config(yscrollcommand=yScrollbar.set)
  yScrollbar.config(command=canvas.yview)

  # plug in the figure
  figAgg = FigureCanvasTkAgg(figure, canvas)
  mplCanvas = figAgg.get_tk_widget()

  # and connect figure with scrolling region
  cwid = canvas.create_window(0, 0, window=mplCanvas, anchor=Tkconstants.NW)
  changeSize(figure, 1)


def changeSize(figure, factor):
  global canvas, mplCanvas, interior, interior_id, frame, cwid
  oldSize = figure.get_size_inches()
  print("old size is", oldSize)
  figure.set_size_inches([factor * s for s in oldSize])
  wi,hi = [i*figure.dpi for i in figure.get_size_inches()]
  print("new size is", figure.get_size_inches())
  print("new size pixels: ", wi,hi)
  mplCanvas.config(width=wi, height=hi)
  canvas.itemconfigure(cwid, width=wi, height=hi)
  canvas.config(scrollregion=canvas.bbox(Tkconstants.ALL),width=200,height=200)

  #figure.tight_layout() # matplotlib > 1.1.1
  #figure.subplots_adjust(left=0.2, bottom=0.15, top=0.86)
  figure.canvas.draw()

if __name__ == "__main__":
  global root, figure
  root = Tk()
  root.rowconfigure(1, weight=1)
  root.columnconfigure(1, weight=1)

  frame = Frame(root)
  frame.grid(column=1, row=19, sticky=Tkconstants.NSEW)
  frame.rowconfigure(1, weight=1)
  frame.columnconfigure(1, weight=1)

  fig, ax = plt.subplots(1,9, sharex=True, figsize=(30,3), squeeze=False)

  X=[1, 2, 3, 4, 5, 6, 7, 8]
  y=[5, 6, 1, 3, 8, 9, 3, 5]

  plot = 0
  for j in range(1,10):
      ax[0,plot].plot(X,y)
      plot = plot + 1

  #tz = figure.text(0.5,0.975,'The master title',horizontalalignment='center', verticalalignment='top')
  addScrollingFigure(fig, frame)



  # figure = plt.figure(dpi=150, figsize=(2,2))
  # ax = figure.add_subplot(111)
  # ax.plot(range(10), [math.sin(x) for x in range(10)])


  root.mainloop()