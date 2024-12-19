import logic
import os
import joblib

import customtkinter as ctk 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import filedialog
from tksheet import Sheet

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("960x600")
        self.title("LIB Recycling Helper")
        self.iconbitmap("icon.ico")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.navigation = NavigationFrame(self, controller=self)
        self.navigation.grid(row=0, column=0, sticky="nesw")

        self.summaryFrame   = FrameSummary(self, controller=self)
        self.impactFrame    = FrameImpact(self, controller=self)
        self.predsFrame     = FramePreds(self, controller=self)
        self.inputsFrame    = FrameInputs(self, controller=self)
        self.optFrame       = FrameOpt(self, controller=self)

        self.optFrame.grid(row=0, column=1, sticky="nsew")
        self.inputsFrame.grid(row=0, column=1, sticky="nsew")
        self.predsFrame.grid(row=0, column=1, sticky="nsew")
        self.impactFrame.grid(row=0, column=1, sticky="nsew")
        self.summaryFrame.grid(row=0, column=1, sticky="nsew")

# Setup navigation bar on the lefthand side of the window
class NavigationFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(fg_color="grey76")
        # self.configure(border_width=2)

        # Title
        title = ctk.CTkLabel(self,
                             text="LIB RECYCLING",
                             font=("Helvetica", 18, 'bold'))
        title.grid(row=0, column=0, padx=(20, 20),pady=(10,0))

        # Add buttons to access the different steps
        btnOptions = ctk.CTkButton(self, 
                                        text="0. Intro",
                                        command=self.introEvent,
                                        font=("Helvetica", 14))
        btnOptions.grid(row=1, column=0, padx=(20, 20), pady=(10,10))

        btnInputs = ctk.CTkButton(self, 
                                        text="1. Inputs",
                                        command=self.inputsEvent,
                                        font=("Helvetica", 14))
        btnInputs.grid(row=2, column=0, padx=(20, 20), pady=(10,10))

        btnPred = ctk.CTkButton(self, 
                                        text="2. Predictions",
                                        command=self.predEvent,
                                        font=("Helvetica", 14))
        btnPred.grid(row=3, column=0, padx=(20, 20), pady=(10,10))

        btnImpact = ctk.CTkButton(self, 
                                        text="3. Eco analysis",
                                        command=self.impactEvent,
                                        font=("Helvetica", 14))
        btnImpact.grid(row=4, column=0, padx=(20, 20), pady=(10,10))

        btnSummary = ctk.CTkButton(self, 
                                        text="4. Summary",
                                        command=self.summaryEvent,
                                        font=("Helvetica", 14))
        btnSummary.grid(row=5, column=0, padx=(20, 20), pady=(10,10))

    # These fucntions are called when the buttons above are pressed
    def introEvent(self):
        print("\nINTRODUCTION")
        self.controller.optFrame.tkraise()
    def inputsEvent(self):
        print("\nINPUTS AND PREDICTIONS")
        self.controller.inputsFrame.tkraise()
    def predEvent(self):
        print("\nPREDICTIONS")
        self.controller.predsFrame.tkraise()
    def impactEvent(self):
        print("\nIMPACTS")
        self.controller.impactFrame.tkraise()
    def summaryEvent(self):
        print("\nSUMMARY")
        self.controller.summaryFrame.tkraise()

# Setup the different windows that appear in the right side of the window
class FrameOpt(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self, 
                                       text="OPTIONS HERE")
        title.grid(row=0, column=0)

class FrameInputs(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        # Setup the grid layout for this frame
        self.grid_columnconfigure((0,1,2,3), weight=1, uniform=1)
        self.grid_rowconfigure((0,1,2,4,5,6,7,8,9), weight=0)
        self.grid_rowconfigure(3,weight=1)

        # LABEL: Title of the window
        title = ctk.CTkLabel(self, text="INPUTS AND PREDICTIONS", font=("Helvetica", 16, 'bold'))
        title.grid(row=0, column=0, columnspan=4, pady=(10,0))

        # LABEL: Subtitle of the window
        subtitle = ctk.CTkLabel(self, text="Predictions are updated when a new file is imported")
        subtitle.grid(row=1, column=0, columnspan=4, pady=(10,0))
        
        ### FILE IMPORT AND TABLE GENERATION ###
        # LABEL: Explain browse button
        self.pickFileLbl = ctk.CTkLabel(self, text="Conditions to process:", font=("Helvetica", 14, 'bold'))
        self.pickFileLbl.grid(row=2, column=0, sticky="nesw", pady=10, padx=10)
        
        # BUTTON: File picking
        self.file_path = ""
        pick_file_button = ctk.CTkButton(self, text="Browse...", command=lambda:self.pick_file(sheet))
        pick_file_button.grid(row=2, column=1, padx=(5,5), pady=(10,))

        # LABEL: Current file name
        self.fileNameLbl = ctk.CTkLabel(self, text='sample.txt')
        self.fileNameLbl.grid(row=2, column=2, columnspan=2, sticky='w', pady=10, padx=10)

        # TKSHEET: Display conditions and preds from the default set
        lst_data = genTable(X_test, y_pred_mu)
        sheet = dfTable(self, lst_data, heathers)
        sheet.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=5, padx=20)

        ### PLOTTING SECTION ###
        # LABEL:   Conditions vs YIELDS
        yieldsLbl = ctk.CTkLabel(self, text='Yields:', font=("Helvetica", 14, 'bold'))
        yieldsLbl.grid(row=4, column=0, sticky='nsew', pady=10, padx=10)

        # BUTTONS: Conditions vs YIELDS
        timeBtn = ctk.CTkButton(self, text='x vs time', command=pltTime)
        timeBtn.grid(row=4, column=1, pady=10, padx=10)

        tempBtn = ctk.CTkButton(self, text='x vs Temp', command=pltTemp)
        tempBtn.grid(row=4, column=2, pady=10, padx=10)

        SLBtn   = ctk.CTkButton(self, text='x vs S/L',  command=pltSL)
        SLBtn.grid(row=4, column=3, pady=10, padx=10)

        acidCBtn = ctk.CTkButton(self, text='x vs [acid]', command=pltAcidC)
        acidCBtn.grid(row=5, column=1, pady=5, padx=10)

        peroxBtn = ctk.CTkButton(self, text='x vs [H2O2]', command=pltPerox)
        peroxBtn.grid(row=5, column=2, pady=5, padx=10)

        acidTypeBtn = ctk.CTkButton(self, text='x vs acid', command=pltAcidType)
        acidTypeBtn.grid(row=5, column=3, pady=5, padx=10)

        # LABEL:   Selectivities
        selctLbl = ctk.CTkLabel(self, text='Selectivities:', font=("Helvetica", 14, 'bold'))
        selctLbl.grid(row=6, column=0, sticky='nsew', pady=10, padx=10)

        # BUTTONS: Selectivities
        heatBtn = ctk.CTkButton(self, text='Heatmap', command=hmapSelect)
        heatBtn.grid(row=6, column=1, pady=5, padx=10)

        enrichBtn = ctk.CTkButton(self, text='Enrichment factor', command=pltEF)
        enrichBtn.grid(row=6, column=2, pady=5, padx=10)

        # LABEL:   Stat. Differences
        diffsLbl = ctk.CTkLabel(self, text='Statistical differences:', font=("Helvetica", 14, 'bold'))
        diffsLbl.grid(row=7, column=0, sticky='nsew', pady=10, padx=10)

        # BUTTONS: Stat. Differences
        diffsBtn =  ctk.CTkButton(self, text='Compare conditions', command = pltStatDiff)
        diffsBtn.grid(row=7, column=1, pady=5, padx=10)

        # LABEL:  Export all plots to .png
        saveAllLbl = ctk.CTkLabel(self, text='Export all as PNG:', font=("Helvetica", 14, 'bold'))
        saveAllLbl.grid(row=7, column=2, sticky='nsew', pady=10, padx=10)

        # BUTTON:  Export all plots to .png
        saveAllBtn = ctk.CTkButton(self, text='Save all', command=saveAllPlts)
        saveAllBtn.grid(row=7, column=3, pady=5, padx=10)

    # This function handles the logic behind the browse button
    def pick_file(self, sheet):
        """Opens a file dialog for .xlsx files, stores the path, and updates the label."""
        # Get the directory of the current .py file
        default_directory = os.path.dirname(os.path.abspath(__file__))

        # Open a file dialog to pick a new .xlsx
        self.file_path = filedialog.askopenfilename(
            initialdir=default_directory,
            title="Select Excel File containing the conditions for prediction",
            filetypes=(("Excel files", "*.xlsx *.xls *.ods"), ("All files", "*.*"))
        )

        # Checks if file_path has been set. 
        # Only relevant in case the browse window is closed before picking a file.
        if self.file_path:
            self.file_name = os.path.basename(self.file_path)  # Extract file name
            self.fileNameLbl.configure(text=f"{self.file_name}")
            global X_test, y_pred_mu, y_pred_std
            # Import data
            X_test,_ = logic.importdata(self.file_path)

            # Compute predictions
            y_pred_mu, y_pred_std = predict(X_test, X_train, y_train)

            # Update table with the results
            lst_data = genTable(X_test, y_pred_mu)

            sheet.destroy()
            sheet = dfTable(self, lst_data, heathers)

            # Update table with new values
            sheet.grid(row=3, column=0, columnspan=4, sticky="nsew", pady=0, padx=20)

class FramePreds(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        
        # Setup the grid for this frame
        self.grid_columnconfigure((0,1), weight=1, uniform="group1")
        self.grid_rowconfigure((0,1,2,4), weight=0)
        self.grid_rowconfigure(3,weight=1)

        # Title of the window
        title = ctk.CTkLabel(self, text="ML RESULTS HERE", font=("Helvetica", 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=(10,0))
    '''
        # Button to generate predictions
        predBtnLabel = ctk.CTkLabel(self, text="Click this button to get predictions")
        predBtnLabel.grid(row=1, column=0, pady=(10,0), sticky='e')
        predButton = ctk.CTkButton(self, text='Generate', command=self.btnPredict)
        predButton.grid(row=1, column=1, pady=(10,0), padx=10, sticky='w')

        # Labels for the inputs and results tables
        inputLabel = ctk.CTkLabel(self, text='Inputs')
        inputLabel.grid(row=2, column=0, columnspan=2, pady=10, padx=10)
        # predLabel = ctk.CTkLabel(self, text='Predictions')
        # predLabel.grid(row=2, column=1, pady=10)

        # # Inputs table
        headers = ["inputNi", 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', '[H2O2]', 'S/L (g/L)', 't (min)']
        
        global results
        lst_data = results.values.tolist()
        # # headers = X_test.columns.tolist()
        self.inSheet = dfTable(self, lst_data, headers)
        self.inSheet.grid(row=3, column=0, columnspan=2, padx=20, pady=(0,10), sticky='nsew')

        # Plot buttons
        plotLbl = ctk.CTkLabel(self, text='Plotting tools', font=("Helvetica", 12, 'bold'))
        plotLbl.grid(row=4, column=0, columnspan=2, padx=10, pady=(0,5))
        
        # kinetics button
        plotKinBtn = ctk.CTkButton(self, text="x vs time", command=self.pltKinetics)
        plotKinBtn.grid(row=5, column=0, padx=10, pady=(0,10))

        # barchart button
        plotBarBtn = ctk.CTkButton(self, text="Barchart", command=self.pltBars)
        plotBarBtn.grid(row=5, column=1, padx=10, pady=(0,10))

    def btnPredict(self):
        print('Calculate predictions')
        global results, y_pred_std

        #Update inputs table before anything else
        lst_data = X_test.values.tolist()
        self.inSheet.set_sheet_data(data=lst_data, reset_col_positions=False)
        # global X_test

        # Generate predictions
        expTest_X  = logic.genFeatures(X_test, X_train)
        y_test_minus_y_pred = mdl[0].predict(expTest_X)

        self.y_pred_mu, self.y_std = logic.twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

        # Display results table
        # lst_data = list(np.around(self.y_pred_mu.tolist(),4))

        # headers = ['Li', 'Ni', 'Mn', 'Co']
        # sheet = dfTable(self, lst_data, headers)
        # sheet.grid(row=3, column=1, padx=(5,20), pady=(0,10), sticky='nsew')

        #Update results dataframe with results

        
        # results = X_test
        lb = ['Li', 'Ni', 'Mn', 'Co']

        for i in range(self.y_pred_mu.shape[1]):
            results[lb[i]] = self.y_pred_mu[:,i]
            y_pred_std[lb[i]] =self.y_std[:,i]


        lst_data = results.round(4).values.tolist()
        headers = ["inputNi", 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', '[H2O2]', 'S/L (g/L)', 't (min)', 'Li', 'Ni', 'Mn', 'Co']
        
        # Redraw table
        self.inSheet = dfTable(self, lst_data, headers)
        self.inSheet.grid(row=3, column=0, columnspan=2, padx=20, pady=(0,10), sticky='nsew')

        # Highlight results 
        self.inSheet.highlight_columns([9, 10, 11, 12], bg='#fcf4e6')

    def pltKinetics(self): #UPDATE TO USE THE RESULTS DF
        # Check if predictions have been generated
        if not hasattr(self, 'y_pred_mu'):
            print("Error: The predictions haven't been calculated yet")
            return
        
        xx = X_test.loc[:,'time']
        yy = self.y_pred_mu
        lb = ['Li', 'Ni', 'Mn', 'Co']
        plt.close('Kinetics')
        plt.figure('Kinetics',figsize=(6, 4))  

        for i in range(yy.shape[1]):
            plt.scatter(xx, yy[:,i], label=lb[i])

        plt.xlabel('time (min)')
        plt.ylabel('Leaching')
        plt.ylim(0,1)
        plt.legend(loc='best')
        print('Plot: kinetics scatter')
        plt.show()

    def pltBars(self):
        global results, y_pred_std
        # Check if predictions have been generated
        if not hasattr(self, 'y_pred_mu'):
            print("Error: The predictions haven't been calculated yet")
            return

        df = results
        columns_to_plot = ['Li', 'Ni', 'Mn', 'Co']
        # title="Grouped Bar Chart"
        xlabel='Condition index'
        ylabel='Leaching'

        n_groups = len(df)
        bar_width = 0.8 / len(columns_to_plot)  # Adjust bar width based on the number of groups
        index = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size as needed

        for i, column in enumerate(columns_to_plot):
            values = df[column].values
            position = index + i * bar_width
            err = y_pred_std[column].values
            rects = ax.bar(position, 
                           values, 
                           bar_width, 
                           yerr=err,
                           ecolor='0.3',
                           capsize=2,
                           label=column)

        # Add some text for labels, title and axes ticks
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_title(title)
        ax.set_ylim([0, 1])
        ax.set_xticks(index + bar_width * (len(columns_to_plot) - 1) / 2) # Center the x ticks
        ax.set_xticklabels(df.index+1)#, rotation=45, ha='right') # Rotate x-axis labels if needed
        ax.legend()

        fig.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()'''

class FrameImpact(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self,
                             text="ECO RESULTS HERE")
        title.grid(row=0, column=0)

class FrameSummary(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self,
                             text="SUMMARY/EXPORT HERE")
        title.grid(row=0, column=0)

def dfTable(parent, tableData: list, tableHeaders: list) -> Sheet:
    """Function that formats TKSheets for displaying conditions and yields"""
    sheet = Sheet(parent, data=tableData, header=tableHeaders)
    sheet.enable_bindings()
    sheet.disable_bindings('edit_cell')
    sheet.disable_bindings('paste')
    sheet.disable_bindings('cut')
    sheet.disable_bindings('delete')
    sheet.disable_bindings("rc_insert_column")
    sheet.disable_bindings("rc_delete_column")
    sheet.disable_bindings("rc_insert_row")
    sheet.disable_bindings("rc_delete_row")
    # sheet.font(("Helvetica", 12, "normal"))
    # sheet.header_font(("Helvetica", 12, "bold"))
    sheet.table_align("right")
    sheet.index_align("center")
    sheet.set_all_cell_sizes_to_text()

    # Highlight results 
    sheet.highlight_columns([9, 10, 11, 12], bg='#e2f2e3')
    return sheet

def predict(X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    """Outputs the prediction average and standard dev as Pandas DataFrames"""
    print("Processing data")
    expTest_X  = logic.genFeatures(X_test, X_train)
    y_test_minus_y_pred = mdl.predict(expTest_X)

    y_pred_mu, y_pred_std = logic.twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

    cols = ['Li', 'Ni', 'Mn', 'Co']

    y_pred_mu  = pd.DataFrame(data=y_pred_mu,  columns=cols)
    y_pred_std = pd.DataFrame(data=y_pred_std, columns=cols)

    return y_pred_mu, y_pred_std

def genTable(X_test: pd.DataFrame, y_pred_mu: pd.DataFrame) -> list:
    """Generates the data to be displayed in tables by combining the conditions and predicted yields"""
    temp = X_test.copy()
    cols_2_copy = ['Li', 'Ni', 'Mn', 'Co']

    for i, col in enumerate(cols_2_copy):
        temp[col] = y_pred_mu[col]

    # Use a tksheet to display the data
    temp = np.around(temp, 4)
    lst_data = temp.values.tolist()

    return lst_data

def pltTime():
    print('* Plot: yield vs time scatter')

    xx = X_test['time']
    yy = y_pred_mu
    lb = ['Li', 'Ni', 'Mn', 'Co']

    plt.close('Kinetics')
    plt.figure('Kinetics',figsize=(6, 4))  

    for i,col in enumerate(lb):
        plt.scatter(xx, yy[col], label=lb[i])

    plt.xlabel('time (min)')
    plt.ylabel('Leaching')
    plt.ylim(0,1)
    plt.legend(loc='best')
    plt.show()

def pltTemp():
    print('* Plot: yield vs temperature scatter')

def pltSL():
    print('* Plot: yield vs S/L scatter')

def pltAcidC():
    print('* Plot: yield vs [acid] scatter')

def pltPerox():
    print('* Plot: yield vs [H2O2] scatter')

def pltAcidType():
    print('* Plot: yield vs acid type bar chart')

def hmapSelect():
    print('* Plot: Heatmap of selectivities')

def pltEF():
    print('* Plot: Enrichment factor bar chart')

def pltStatDiff():
    print('* Plot: Statistical differences')

def saveAllPlts():
    print('> Save all plots as .png')

# Load pre-trained PD-GBR model
mdl,_,_ = joblib.load('model/PGBR.gz')

# Load training data
X_train = joblib.load('model/xtrain.gz')
y_train = joblib.load('model/ytrain.gz')

# Import sample data
X_test,_ = logic.importdata("sample.xlsx")

# Process the sample data
y_pred_mu, y_pred_std = predict(X_test, X_train, y_train)

heathers = ['inputNi', 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', 
            '[H2O2]', 'S/L (g/L)', 't (min)', 'Li', 'Ni', 'Mn', 'Co']


app = App()
app.mainloop()