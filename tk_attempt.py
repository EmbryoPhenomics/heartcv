from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import cv2
import vuba
import heartcv as hcv

class FeetToMeters:

    def __init__(self, root):
        self.root = root

        self.root.title("HeartCV")

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Button(mainframe, text='Open Video', command=self.open_file).grid(column=1, row=1, sticky=W)
        ttk.Button(mainframe, text='Load Example', command=self.open_file).grid(column=2, row=1, sticky=E)

        ttk.Scale(mainframe, from_=2, to=64, orient='horizontal', showvalue=0).grid(column=1, row=4, sticky=W)
       
        ttk.Button(mainframe, text='Calculate', command=self.calculate).grid(column=1, row=5, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        # feet_entry.focus()
        # self.root.bind("<Return>", self.calculate)

    def open_file(self):
        self.filename = filedialog.askopenfilename()
        print(self.filename)

    def calculate(self, *args):
        video = vuba.Video(self.filename)
        frames = video.read(0, 300, grayscale=True)

        ept = hcv.epts(frames, fs=video.fps, binsize=int(self.binsize.get()))
        power_map = hcv.spectral_map(ept, frequencies=float(self.freq.get()))

        power_map = cv2.resize(power_map, video.resolution)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)

        for ax in (ax1, ax2):
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.imshow(vuba.take_first(frames), cmap='gray')
        ax2.imshow(power_map)

        ax1.set_title('Input Video')
        ax2.set_title('Spectral Heatmap')

        canvas = FigureCanvasTkAgg(fig, master=self.root).get_tk_widget().grid(column=3, row=0, sticky=E)
        canvas.draw()

root = Tk()
FeetToMeters(root)
root.mainloop()