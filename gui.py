import logic
import os
import joblib

import customtkinter as ctk 
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog
from tksheet import Sheet

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# X_test = False;

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x600")
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
                                        text="0. Options",
                                        command=self.optnEvent,
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
    def optnEvent(self):
        print("\nOptions frame open")
        self.controller.optFrame.tkraise()
    def inputsEvent(self):
        print("\nInputs frame open")
        self.controller.inputsFrame.tkraise()
    def predEvent(self):
        print("\nPredictions frame open")
        self.controller.predsFrame.tkraise()
    def impactEvent(self):
        print("\nImpacts frame open")
        self.controller.impactFrame.tkraise()
    def summaryEvent(self):
        print("\nSummary frame open")
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

# This window handles the files containing the conditions to predict
class FrameInputs(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Setup the grid for this frame
        self.grid_columnconfigure((0,1), weight=1, uniform="group1")
        self.grid_rowconfigure((0,1,2,3), weight=0)
        self.grid_rowconfigure(4,weight=1)

        # Title of the window with some explanation
        title = ctk.CTkLabel(self, text="INPUTS TITLE HERE", font=("Helvetica", 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=(10,0))
        subtitle = ctk.CTkLabel(self, text="These are the conditions for which the yields will be predicted")
        subtitle.grid(row=1, column=0, columnspan=2, pady=(10,0))

        # Label and button to allow picking of a test data file
        self.file_path = ""
        pick_file_button = ctk.CTkButton(
            master=self, 
            text="Browse...", 
            command=lambda:self.pick_file(sheet))
        pick_file_button.grid(row=2, column=1, sticky="w", padx=(5,5), pady=(10,))

        self.fileLabel = ctk.CTkLabel(self, text="No file selected.", font=("Helvetica", 14, 'bold'))
        self.fileLabel.grid(row=2, column=0, sticky="e", padx=(5,5), pady=(10,))

        # A simple label for the table
        tableLabel=ctk.CTkLabel(self, text="Conditions:", font=("Helvetica", 14, 'bold'))
        tableLabel.grid(row=3, column=0, columnspan=2, sticky="sw", padx=10)

        # Use a tksheet to display the input data
        lst_data = X_test.values.tolist()
        # headers = X_test.columns.tolist()

        headers = ["inputNi", 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', '[H2O2]', 'S/L (g/L)', 't (min)']

        sheet = dfTable(self, lst_data, headers)
        sheet.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0,20), padx=(20,20))

    # This function handles the logic behind the browse button
    def pick_file(self, sheet):
        """Opens a file dialog for .xlsx files, stores the path, and updates the label."""
        # Get the directory of the current .py file
        default_directory = os.path.dirname(os.path.abspath(__file__))
        self.file_path = filedialog.askopenfilename(
            initialdir=default_directory,
            title="Select Excel File containing the conditions for prediction",
            filetypes=(("Excel files", "*.xlsx *.xls *.ods"), ("All files", "*.*"))
        )

        # Checks if file_path has been set. Only relevant in case the browse window is closed before picking a file.
        if self.file_path:
            self.file_name = os.path.basename(self.file_path)  # Extract file name
            self.fileLabel.configure(text=f"{self.file_name}")
            
            global X_test
            X_test,_ = logic.importdata(self.file_path)

            # Try using a tksheet
            lst_data = X_test.values.tolist()
            # headers = ["inputNi", 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', '[H2O2] (wt.%)', 'S/L (g/L)', 't (min)']
            # Update table with new values
            sheet.set_sheet_data(data=lst_data, reset_col_positions=False) 
            # sheet = self.dfTable(lst_data, headers)
            # sheet.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0,20), padx=(20,20))
      
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

        # Button to generate predictions
        predBtnLabel = ctk.CTkLabel(self, text="Click this button to get predictions")
        predBtnLabel.grid(row=1, column=0, pady=(10,0), sticky='e')
        predButton = ctk.CTkButton(self, text='Generate', command=self.btnPredict)
        predButton.grid(row=1, column=1, pady=(10,0), padx=10, sticky='w')

        # Labels for the inputs and results tables
        inputLabel = ctk.CTkLabel(self, text='Inputs')
        inputLabel.grid(row=2, column=0, pady=10)
        predLabel = ctk.CTkLabel(self, text='Predictions')
        predLabel.grid(row=2, column=1, pady=10)

        # Inputs table
        headers = ["inputNi", 'inputMn', 'inputCo', 'T (°C)', 'pKa1', '[acid] (M)', '[H2O2]', 'S/L (g/L)', 't (min)']
        
        lst_data = X_test.values.tolist()
        # headers = X_test.columns.tolist()
        self.inSheet = dfTable(self, lst_data, headers)
        self.inSheet.grid(row=3, column=0, padx=(20,5), pady=(0,10), sticky='nsew')

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

        #Update inputs table before anything else
        lst_data = X_test.values.tolist()
        self.inSheet.set_sheet_data(data=lst_data, reset_col_positions=False)
        # global X_test

        # Generate predictions
        expTest_X  = logic.genFeatures(X_test, X_train)
        y_test_minus_y_pred = mdl[0].predict(expTest_X)

        self.y_pred_mu, y_pred_std = logic.twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

        # Display results table
        lst_data = list(np.around(self.y_pred_mu.tolist(),4))

        headers = ['Li', 'Ni', 'Mn', 'Co']
        sheet = dfTable(self, lst_data, headers)
        sheet.grid(row=3, column=1, padx=(5,20), pady=(0,10), sticky='nsew')

        #Make results dataframe
        global results
        results = X_test
        lb = ['Li', 'Ni', 'Mn', 'Co']

        for i in range(self.y_pred_mu.shape[1]):
            results[lb[i]] = self.y_pred_mu[:,i]

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
        # TODO: Implement having all of the metals side by side - Optionally, print 4 subplots instead
        global results
        # Check if predictions have been generated
        if not hasattr(self, 'y_pred_mu'):
            print("Error: The predictions haven't been calculated yet")
            return
        
        plt.close('Bars')
        plt.figure('Bars',figsize=(6, 4)) 

        xx = results.index
        yy = results['Li']
        plt.bar(xx, yy)
        yy = results['Ni']
        plt.bar(xx, yy)


        
        # lb = ['Li', 'Ni', 'Mn', 'Co']
        # plt.close('Kinetics')
        # plt.figure('Kinetics',figsize=(6, 4))  

        # for i in range(yy.shape[1]):
        #     plt.scatter(xx, yy[:,i], label=lb[i])

        # plt.legend(loc='best')
        plt.xticks(xx, xx)
        # plt.xlabel("Index")
        plt.ylabel("Values")
        plt.ylim(0,1)
        plt.title("Bar Chart of Values")
        print('Plot: Bar chart')
        plt.show()


        

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

def dfTable(parent, tableData, tableHeaders):
    """Function that handles displaying dataFrames for consistency"""
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
    return sheet
  
# Load pre-trained PD-GBR model
mdl = joblib.load('model/PGBR.gz')

# Load training data
X_train = joblib.load('model/xtrain.gz')
y_train = joblib.load('model/ytrain.gz')

# Import sample data
X_test,_ = logic.importdata("sample.xlsx")

app = App()
app.mainloop()