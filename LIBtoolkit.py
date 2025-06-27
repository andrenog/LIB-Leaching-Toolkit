import logic
import os
import joblib
import datetime
import warnings
import webbrowser

import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from tkinter import filedialog
from tkinter import messagebox
from tksheet import Sheet
from itertools import combinations
from PIL import Image


class App(ctk.CTk):
    '''
    Main app window.

    This class initializes the main window, sets up the layout,
    and manages the frames that display content for the application.
    '''
    def __init__(self):
        '''
        Initializes the App class.
        '''
        super().__init__()
        self.geometry("960x600")
        self.title("LIB Leaching Toolkit")
        self.iconbitmap(getPath("icon.ico"))

        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.navigation = NavigationFrame(self, controller=self)
        self.navigation.grid(row=0, column=0, sticky="nesw")

        self.calcsFrame = FrameCalcs(self, controller=self)
        self.dataFrame = FrameData(self, controller=self)
        self.homeFrame = FrameHome(self, controller=self)

        self.homeFrame.grid(row=1, column=0, sticky="nsew")
        self.dataFrame.grid(row=1, column=0, sticky='nsew')
        self.calcsFrame.grid(row=1, column=0, sticky="nsew")


class NavigationFrame(ctk.CTkFrame):
    '''
    Navigation bar.

    This class creates a frame containing the navigation elements for the
    application, including buttons to switch between different sections
    and export options.
    '''
    def __init__(self, parent, controller):
        '''
        Initializes the navigation bar.
        '''
        super().__init__(parent)
        self.controller = controller
        self.configure(fg_color="grey76")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1, 2, 3, 4, 5), weight=0)

        # Title
        title = ctk.CTkLabel(self,
                             text="LIB Leaching Toolkit",
                             font=("Helvetica", 20, 'bold'))
        title.grid(row=0, column=0, padx=10, pady=10)

        # Add buttons to access the different steps
        btnHome = ctk.CTkButton(self,
                                text="Home",
                                command=self.homeEvent,
                                font=("Helvetica", 14))
        btnHome.grid(row=0, column=1, padx=10)

        btnData = ctk.CTkButton(self,
                                text='Input generation',
                                command=self.dataEvent,
                                font=("Helvetica", 14))
        btnData.grid(row=0, column=2, padx=10)

        btnInputs = ctk.CTkButton(self,
                                  text="Analysis",
                                  command=self.calcsEvent,
                                  font=("Helvetica", 14))
        btnInputs.grid(row=0, column=3, padx=10)


    # These functions are called when the buttons above are pressed
    def homeEvent(self):
        '''
        Switches to the Home frame
        '''
        self.controller.homeFrame.tkraise()

    def dataEvent(self):
        '''
        Switches to the Data frame
        '''
        self.controller.dataFrame.tkraise()

    def calcsEvent(self):
        '''
        Switches to the Calculations frame
        '''
        self.controller.calcsFrame.tkraise()

class FrameHome(ctk.CTkFrame):
    '''
    Setup the Home frame
    '''
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_columnconfigure(0, weight=1)

        img = ctk.CTkImage(light_image=Image.open(getPath('data/home.png')),
                           size=(960, 540))
        imgLbl = ctk.CTkLabel(self, text='', image=img)
        imgLbl.grid(row=0, column=0, sticky='nsew')

        imgLbl.bind("<Button-1>", self.open_github)

    def open_github(self, event):
        '''
        Opens the GitHub repository in the default web browser
        when the image is clicked.
        '''
        url = "https://github.com/andrenog/LIB-Leaching-Toolkit"
        print(f"Opening {url} in web browser...")
        webbrowser.open(url, new=2)  # Open in a new tab if possible

class FrameData(ctk.CTkFrame):
    '''
    Setup the Data frame
    '''
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller # Store the controller

        self.grid_columnconfigure((0,1,2), weight=0)
        self.grid_columnconfigure(3, weight=1)

        # Make placeholders for the widgets
        self.tempLbl = None # Temperature label
        self.inTemp = None # Temperature input

        self.acidLbl = None # Acid label
        self.inAcid = None # Acid input
        self.acid_label = None # Acid label

        self.acidCLbl = None # Acid concentration label
        self.inAcidC = None # Acid concentration input

        self.peroxLbl = None # H2O2 concentration label
        self.inPerox = None # H2O2 concentration input

        self.SLLbl = None # S/L ratio label
        self.inSL = None # S/L ratio input

        self.timeLbl = None # Time label
        self.inTime = None # Time input

        # LABEL: Fixed condition
        self.fixLbl = ctk.CTkLabel(self, text='Select condition to vary:', font=('Helvetica', 14, 'bold'))
        self.fixLbl.grid(row=0, column=0, columnspan=2,sticky='nse', pady=5, padx=10)
        self.condFix_var = ctk.StringVar(value='Select') # set initial value
        
        options = ['T (°C)', '[acid] (M)', '[H2O2]', 'S/L (g/L)', 't (min)']
        dropVaryCond = ctk.CTkOptionMenu(master=self,
                                     values=options,
                                     width=100,
                                     command=self.fixMenu_callback,
                                     variable=self.condFix_var)
        
        dropVaryCond.grid(row=0, column=2, pady=5, padx=10)

        # CONDITION to vary
        self.minLbl = ctk.CTkLabel(self, text='Minimum: ')
        self.maxLbl = ctk.CTkLabel(self, text='Maximum: ')
        self.stepLbl = ctk.CTkLabel(self, text='Steps: ')

        self.minEntry = ctk.CTkEntry(self,placeholder_text='min', width=100,
                                 validatecommand=lambda: validate(self.minEntry),
                                 validate='focusout')
        self.maxEntry = ctk.CTkEntry(self,placeholder_text='max', width=100,
                                 validatecommand=lambda: validate(self.maxEntry),
                                 validate='focusout')
        self.stepEntry = ctk.CTkEntry(self,placeholder_text='steps', width=100,
                                 validatecommand=lambda: validate(self.stepEntry, 'int'),
                                 validate='focusout')

        self.line = ctk.CTkProgressBar(self, orientation='horizontal', height=2)
        self.line.set(1)

        genInputBtn = ctk.CTkButton(self, text='Generate table', width=100,
                                    command=self.genInputTable)
        genInputBtn.grid(row=14, column=1, padx=10, pady=5)

        self.hide_interval_widgets()

    def fixMenu_callback(self, choice):
        print('Condition to vary: ', choice)

        global variable_to_vary
        variable_to_vary = choice

        self.show_interval_widgets()

        # Pick NMC type
        NMCLbl = ctk.CTkLabel(self, text='NMC type', font=('Helvetica', 14, 'bold'))
        NMCLbl.grid(row=4, column=0, padx=10, pady=5, columnspan=3)

        self.inNi = ctk.CTkEntry(self,placeholder_text='Ni', width=100,
                                 validatecommand=lambda: validate(self.inNi),
                                 validate='focusout')
        
        NiLbl = ctk.CTkLabel(self, text='Ni')
        NiLbl.grid(row=5, column=0, padx=10, pady=0)
        MnLbl = ctk.CTkLabel(self, text='Mn')
        MnLbl.grid(row=5, column=1, padx=10, pady=0)
        CoLbl = ctk.CTkLabel(self, text='Co')
        CoLbl.grid(row=5, column=2, padx=10, pady=0)

        self.inNi.grid(row=6, column=0, padx=10, pady=5)

        self.inMn = ctk.CTkEntry(self,placeholder_text='Mn', width=100,
                                 validatecommand=lambda: validate(self.inMn),
                                 validate='focusout')
        self.inMn.grid(row=6, column=1, padx=10, pady=5)

        self.inCo = ctk.CTkEntry(self,placeholder_text='Co', width=100,
                                 validatecommand=lambda: validate(self.inCo),
                                 validate='focusout')
        self.inCo.grid(row=6, column=2, padx=10, pady=5)

        fixedVarLbl = ctk.CTkLabel(self, text='Fixed variables', font=('Helvetica', 14, 'bold'))
        fixedVarLbl.grid(row=7, column=0, columnspan=3, padx=10, pady=5)

        def rmWidget(widget):
            if widget is not None:
                widget.grid_remove()
                return None

        # Remove extra fields if they are present
        match choice:
            case 'T (°C)':
                self.tempLbl = rmWidget(self.tempLbl)
                self.inTemp = rmWidget(self.inTemp)
            case 'pKa1':
                self.acidLbl = rmWidget(self.acidLbl)
                self.inAcid = rmWidget(self.inAcid)
                self.acid_label = rmWidget(self.acid_label)
            case '[acid] (M)':
                self.acidCLbl = rmWidget(self.acidCLbl)
                self.inAcidC = rmWidget(self.inAcidC)
            case '[H2O2]':
                self.peroxLbl = rmWidget(self.peroxLbl)
                self.inPerox = rmWidget(self.inPerox)
            case 'S/L (g/L)':
                self.SLLbl = rmWidget(self.SLLbl)
                self.inSL = rmWidget(self.inSL)
            case 't (min)':
                self.timeLbl = rmWidget(self.timeLbl)
                self.inTime = rmWidget(self.inTime)

        # Temperature
        if choice != 'T (°C)':
            if self.tempLbl is None:
                self.tempLbl = ctk.CTkLabel(self, text='T (°C): ')
                self.inTemp = ctk.CTkEntry(self,placeholder_text='T (°C)', width=100,
                                  validatecommand=lambda: validate(self.inTemp),
                                  validate='focusout')
            self.tempLbl.grid(row=8, column=0, sticky='nse', padx=10, pady=5)
            self.inTemp.grid(row=8, column=1, padx=10, pady=5)

        # Acid (pKa1)
        if choice != 'pKa1':
            if self.acidLbl is None:
                self.acidLbl = ctk.CTkLabel(self, text='Acid: ')
                acid_names = acidProps['Acid'].tolist()
                acid_names.append("Custom")
                self.inAcid = ctk.CTkComboBox(self, values=acid_names, width=100, command=self.custom_acid_selected)

            self.inAcid.set(acid_names[0])  # Set initial value

            self.acidLbl.grid(row=9, column=0, sticky='nse', padx=10, pady=5)
            self.inAcid.grid(row=9, column=1, padx=10, pady=5)

        # Acid concentration
        if choice != '[acid] (M)':
            if self.acidCLbl is None:
                self.acidCLbl = ctk.CTkLabel(self, text='[acid] (M): ')
                self.inAcidC = ctk.CTkEntry(self,placeholder_text='[acid] (M)', width=100,
                                  validatecommand=lambda: validate(self.inAcidC),
                                  validate='focusout')
            self.acidCLbl.grid(row=10, column=0, sticky='nse', padx=10, pady=5)
            self.inAcidC.grid(row=10, column=1, padx=10, pady=5)

        # H2O2 concentration
        if choice != '[H2O2]':
            if self.peroxLbl is None:
                self.peroxLbl = ctk.CTkLabel(self, text='[H2O2]: ')
                self.inPerox = ctk.CTkEntry(self,placeholder_text='[H2O2]', width=100,
                                  validatecommand=lambda: validate(self.inPerox),
                                  validate='focusout')
            self.peroxLbl.grid(row=11, column=0, sticky='nse', padx=10, pady=5)
            self.inPerox.grid(row=11, column=1, padx=10, pady=5)

        # S/L ratio
        if choice != 'S/L (g/L)':
            if self.SLLbl is None:
                self.SLLbl = ctk.CTkLabel(self, text='S/L (g/L): ')
                self.inSL = ctk.CTkEntry(self,placeholder_text='S/L (g/L)', width=100,
                                  validatecommand=lambda: validate(self.inSL),
                                  validate='focusout')
            self.SLLbl.grid(row=12, column=0, sticky='nse', padx=10, pady=5)
            self.inSL.grid(row=12, column=1, padx=10, pady=5)

        # time
        if choice != 't (min)':
            if self.timeLbl is None:
                self.timeLbl = ctk.CTkLabel(self, text='t (min): ')
                self.inTime = ctk.CTkEntry(self,placeholder_text='t (min)', width=100,
                                  validatecommand=lambda: validate(self.inTime),
                                  validate='focusout')
            self.timeLbl.grid(row=13, column=0, sticky='nse', padx=10, pady=5)
            self.inTime.grid(row=13, column=1, padx=10, pady=5)

    
    def genInputTable(self):
        print('> Generating input table')
        tab = pd.DataFrame()

        # Get all of the data from the different fields
        # Generate the vector containing the variable to vary
        try:
            intervalMin = float(self.minEntry.get())
            intervalMax = float(self.maxEntry.get())
            intervalStep = int(self.stepEntry.get())
        except ValueError:
            print('!! Please enter valid values for the interval !!')
            messagebox.showerror('Error', 'Please enter valid values for the interval')
            return

        fixed_vars = {
            'T (°C)': 'temp',
            '[acid] (M)': 'acidC',
            '[H2O2]': 'H2O2_conc',
            'S/L (g/L)': 'solidToLiquid',
            't (min)': 'time'
        }

        r = np.linspace(intervalMin, intervalMax, intervalStep)

        notFixVar = self.condFix_var.get()

        tab[fixed_vars[notFixVar]] = r

        # The NMC type is always fixed
        tab['inputNi'] = float(self.inNi.get())
        tab['inputMn'] = float(self.inMn.get())
        tab['inputCo'] = float(self.inCo.get())

        # Get the fixed variables from the user input
        if notFixVar != 'T (°C)':
            tab['temp'] = float(self.inTemp.get())
        if notFixVar != '[acid] (M)':
            tab['acidC'] = float(self.inAcidC.get())
        if notFixVar != '[H2O2]':
            tab['H2O2_conc'] = float(self.inPerox.get())
        if notFixVar != 'S/L (g/L)':
            tab['solidToLiquid'] = float(self.inSL.get())
        if notFixVar != 't (min)':
            tab['time'] = float(self.inTime.get())
        
        # Get the acid properties
        # If the user selected a custom acid, get the properties from the dialog
        if self.inAcid.get() == "Custom":
            print('!! Custom acid selected !!')
        # Else, get the properties from the acidProps DataFrame
        else:
            acid_name = self.inAcid.get()
            acid_row = acidProps[acidProps['Acid'] == acid_name].iloc[0]
            tab['Acid'] = acid_row['Acid']
            tab['pKa1'] = acid_row['pKa1']
            tab['pKa2'] = acid_row['pKa2']
            tab['pKa3'] = acid_row['pKa3']
            tab['nProtons'] = acid_row['nProtons']
            tab['SMILES'] = acid_row['SMILES']
            tab['sLi'] = acid_row['sLi']
            tab['sNi'] = acid_row['sNi']
            tab['sMn'] = acid_row['sMn']
            tab['sCo'] = acid_row['sCo']

        # Order the table to match the order in columns
        tab = tab[columns]

        export_tab = tab.copy()

        print('Export table:')
        print(export_tab.head())

        # Add empty columns for 'xLi', 'xNi', 'xMn', 'xCo'
        for col in ['xLi', 'xNi', 'xMn', 'xCo']:
            export_tab[col] = np.nan

        # Save the export_tab DataFrame as xlsx
        fname = saveAsXLSX(export_tab, 'input_table')
        print('> Save table as ', fname)

        global generated_file_path
        generated_file_path = os.path.join('outputs', fname)

        infoCalc = ctk.CTkLabel(self, text='Table saved. See ANALYSIS section for predictions and plots', font=('Helvetica', 14, 'bold'))
        infoCalc.grid(row=14, column=3, padx=10, pady=5)

        # Update table with the results
        temp = np.around(tab, 4)
        lst_data = temp.values.tolist()
        
        # Check if the global 'sheet' exists and is not None
        if 'inputSheet' in globals() and sheet is not None:
            inputSheet.destroy()
            inputSheet = None  # Reset the global sheet variable



        inputSheet = dfTable(self, lst_data, heathers)

        # Update table with new values
        inputSheet.grid(row=0, column=3, rowspan=12,
                        sticky="nsew", pady=10, padx=10)

        # Trigger file picking in FrameCalcs
        self.controller.calcsFrame.pick_file()  # Call the pick_file function

    def update_acid_label(self, event=None):
        """Updates the acid label based on the selected pKa value."""
        pka = self.inpKa.get()
        try:
            acid_name = pka_map[float(pka)]  # Look up acid name
            self.acid_label.configure(text=f"{acid_name}")
            self.inpKa.configure(border_color='green')
        except KeyError:
            self.acid_label.configure(text="Custom")
        except ValueError:
            self.inpKa.configure(border_color='red')
            return

    def show_interval_widgets(self):
        self.minLbl.grid(row=1, column=0, pady=5, padx=10)
        self.minEntry.grid(row=2, column=0, pady=0, padx=10)
        self.maxLbl.grid(row=1, column=1, pady=5, padx=10)
        self.maxEntry.grid(row=2, column=1, pady=0, padx=10)
        self.stepLbl.grid(row=1, column=2, pady=5, padx=10)
        self.stepEntry.grid(row=2, column=2, pady=0, padx=10)
        self.line.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

    def hide_interval_widgets(self):
        self.minLbl.grid_remove()
        self.minEntry.grid_remove()
        self.maxLbl.grid_remove()
        self.maxEntry.grid_remove()
        self.stepLbl.grid_remove()
        self.stepEntry.grid_remove()
        self.line.grid_remove()
        
    def custom_acid_selected(self, choice):
        if choice == "Custom":
            CustomAcidDialog(self)

class CustomAcidDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.geometry("400x500")
        self.title("Custom Acid Properties")
        self.parent = parent

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Acid Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.acid_name_entry = ctk.CTkEntry(self)
        self.acid_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="pKa1:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.pka1_entry = ctk.CTkEntry(self)
        self.pka1_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="pKa2:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.pka2_entry = ctk.CTkEntry(self)
        self.pka2_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="pKa3:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.pka3_entry = ctk.CTkEntry(self)
        self.pka3_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Num Protons:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.num_protons_entry = ctk.CTkEntry(self)
        self.num_protons_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Smiles String:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.smiles_entry = ctk.CTkEntry(self)
        self.smiles_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Solubility Li:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.solubility_li_entry = ctk.CTkEntry(self)
        self.solubility_li_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Solubility Ni:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        self.solubility_ni_entry = ctk.CTkEntry(self)
        self.solubility_ni_entry.grid(row=7, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Solubility Mn:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        self.solubility_mn_entry = ctk.CTkEntry(self)
        self.solubility_mn_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self, text="Solubility Co:").grid(row=9, column=0, padx=5, pady=5, sticky="e")
        self.solubility_co_entry = ctk.CTkEntry(self)
        self.solubility_co_entry.grid(row=9, column=1, padx=5, pady=5, sticky="w")

        self.save_button = ctk.CTkButton(self, text="Save", command=self.save_properties)
        self.save_button.grid(row=10, column=0, columnspan=2, padx=5, pady=10)

    def save_properties(self):
        acid_name = self.acid_name_entry.get()
        pka1 = self.pka1_entry.get()
        pka2 = self.pka2_entry.get()
        pka3 = self.pka3_entry.get()
        num_protons = self.num_protons_entry.get()
        smiles = self.smiles_entry.get()
        solubility_li = self.solubility_li_entry.get()
        solubility_ni = self.solubility_ni_entry.get()
        solubility_mn = self.solubility_mn_entry.get()
        solubility_co = self.solubility_co_entry.get()

        # Basic validation - check if entries are empty
        if not all([acid_name, pka1, num_protons, solubility_li, solubility_ni, solubility_mn, solubility_co]):
            messagebox.showerror("Error", "Please fill in all the fields.")
            return

        # Store the properties (you might want to store them in a more structured way)
        custom_acid_properties = {
            'Acid': acid_name,
            'pKa1': pka1,
            'pKa2': pka2,
            'pKa3': pka3,
            'nProtons': num_protons,
            'SMILES': smiles,
            'sLi': solubility_li,
            'sNi': solubility_ni,
            'sMn': solubility_mn,
            'sCo': solubility_co
        }

        # Print the properties to the console
        print("Custom Acid Properties:", custom_acid_properties)

        # Optionally, update acidProps DataFrame or other relevant data structures
        # For example:
        # global acidProps
        # new_acid_df = pd.DataFrame([custom_acid_properties])
        # acidProps = pd.concat([acidProps, new_acid_df], ignore_index=True)

        self.destroy()

class FrameCalcs(ctk.CTkFrame):
    '''
    Calculations frame.

    This class creates a frame that allows users to import data,
    generate predictions, visualize results, and analyze
    various aspects of the LIB recycling process.
    '''
    def __init__(self, parent, controller):
        '''
        Initializes Calculations frame.

        Sets up:
            - Title label
            - File selection and import
            - Table display
            - Plotting controls
        '''
        super().__init__(parent)
        # Setup the grid layout for this frame
        self.grid_columnconfigure((0, 1, 2, 3, 4), weight=1, uniform=1)
        self.grid_rowconfigure((0, 1, 2, 4, 5, 6, 7, 8, 9), weight=0)
        self.grid_rowconfigure(3, weight=1)

        # FILE IMPORT
        # LABEL: Explain browse button
        self.pickFileLbl = ctk.CTkLabel(self,
                                        text="1. Select conditions to start:",
                                        font=("Helvetica", 14, 'bold'))
        self.pickFileLbl.grid(row=0, column=0, columnspan=2,
                              sticky="nse", pady=5, padx=10)

        # BUTTON: File picking
        self.file_path = ""
        pick_file_button = ctk.CTkButton(self, text="Browse...",
                                         command=self.pick_file,
                                         fg_color='#2da44e',
                                         hover_color='#2c974b')
        pick_file_button.grid(row=0, column=2, padx=(5, 5), pady=5)

        # LABEL: Current file name
        self.fileNameLbl = ctk.CTkLabel(self, text='')
        self.fileNameLbl.grid(row=0, column=3, columnspan=2,
                              sticky='w', pady=5, padx=10)

        # LABEL: Export
        self.exportLbl = ctk.CTkLabel(self,
                                      text='2. Export data and plots:',
                                      font=('Helvetica', 14, 'bold'))
        self.exportLbl.grid(row=1, column=0, columnspan=2,
                            sticky='nse', pady=(5,10), padx=10)
        
        # BUTTONS: Export
        self.exportDataBtn = ctk.CTkButton(self, text='Export results',
                                           command=self.exportResults)
        self.exportDataBtn.grid(row=1, column=2, padx=5, pady=(5,10))

        self.exportPlotsBtn = ctk.CTkButton(self, text='Export plots',
                                            command=self.exportPlots)
        self.exportPlotsBtn.grid(row=1, column=3, padx=5, pady=(5,10))

        # PLOTTING SECTION
        self.yieldsLbl = ctk.CTkLabel(self, text='Extraction:',
                                 font=("Helvetica", 14, 'bold'))

        self.timeBtn = ctk.CTkButton(self, text='x vs time',
                                command=pltTime)
        self.tempBtn = ctk.CTkButton(self, text='x vs Temp',
                                command=pltTemp)
        self.SLBtn = ctk.CTkButton(self, text='x vs S/L',
                              command=pltSL)
        self.acidCBtn = ctk.CTkButton(self, text='x vs [acid]',
                                 command=pltAcidC)
        self.peroxBtn = ctk.CTkButton(self, text='x vs [H2O2]',
                                 command=pltPerox)
        self.acidTypeBtn = ctk.CTkButton(self, text='x vs acid',
                                    command=pltAcidType)

        self.selctLbl = ctk.CTkLabel(self, text='Selectivities:',
                                font=("Helvetica", 14, 'bold'))
        self.heatBtn = ctk.CTkButton(self, text='Heatmap',
                                command=hmapSelect)
        self.enrichBtn = ctk.CTkButton(self, text='Enrichment factor',
                                  command=pltEF)

        self.impactLbl = ctk.CTkLabel(self, text='Env. impacts:',
                                 font=("Helvetica", 14, 'bold'))
        self.impactBtn = ctk.CTkButton(self, text='EI Heatmap', command=pltEI)

        self.diffsLbl = ctk.CTkLabel(self, text='Stat. differences:',
                                font=("Helvetica", 14, 'bold'))
        self.diffsBtn = ctk.CTkButton(self, text='Compare conditions',
                                 command=pltStatDiff)

        self.costsLbl = ctk.CTkLabel(self, text='Costs:',
                                font=("Helvetica", 14, 'bold'))
        self.costsBtn = ctk.CTkButton(self, text='Costs',
                                 command=pltCosts)

        self.hide_plotting_buttons()

    def show_plotting_buttons(self):
        """Shows relevant plotting buttons based on the file source."""
        global file_source
        self.hide_plotting_buttons()

        # Common buttons
        self.selctLbl.grid(row=6, column=0, sticky='nsew', pady=10, padx=10)
        self.heatBtn.grid(row=6, column=1, pady=5, padx=10)
        self.enrichBtn.grid(row=6, column=2, pady=5, padx=10)
        self.impactLbl.grid(row=6, column=3, sticky='nsew', pady=10, padx=10)
        self.impactBtn.grid(row=6, column=4, pady=5, padx=10)
        self.diffsLbl.grid(row=7, column=0, sticky='nsew', pady=10, padx=10)
        self.diffsBtn.grid(row=7, column=1, pady=5, padx=10)
        self.costsLbl.grid(row=7, column=3, sticky='nsew', pady=10, padx=10)
        self.costsBtn.grid(row=7, column=4, pady=5, padx=10)
        self.yieldsLbl.grid(row=4, column=0, sticky='nsew', pady=10, padx=10)

        if file_source == "generated":
            # Buttons for generated input tables
            if variable_to_vary == 't (min)':
                self.timeBtn.grid(row=4, column=1, pady=10, padx=10)
            elif variable_to_vary == 'T (°C)':
                self.tempBtn.grid(row=4, column=2, pady=10, padx=10)
            elif variable_to_vary == 'S/L (g/L)':
                self.SLBtn.grid(row=4, column=3, pady=10, padx=10)
            elif variable_to_vary == '[acid] (M)':
                self.acidCBtn.grid(row=4, column=4, pady=5, padx=10)
            elif variable_to_vary == '[H2O2]':
                self.peroxBtn.grid(row=5, column=1, pady=5, padx=10)
            elif variable_to_vary == 'pKa1':
                self.acidTypeBtn.grid(row=5, column=2, pady=5, padx=10)

        else:
            # Buttons for browsed files (show all)
            self.timeBtn.grid(row=4, column=1, pady=10, padx=10)
            self.tempBtn.grid(row=4, column=2, pady=10, padx=10)
            self.SLBtn.grid(row=4, column=3, pady=10, padx=10)
            self.acidCBtn.grid(row=4, column=4, pady=5, padx=10)
            self.peroxBtn.grid(row=5, column=1, pady=5, padx=10)
            self.acidTypeBtn.grid(row=5, column=2, pady=5, padx=10)

    def hide_plotting_buttons(self):
        """Hides all plotting buttons."""
        self.timeBtn.grid_remove()
        self.tempBtn.grid_remove()
        self.SLBtn.grid_remove()
        self.acidCBtn.grid_remove()
        self.peroxBtn.grid_remove()
        self.acidTypeBtn.grid_remove()
        self.selctLbl.grid_remove()
        self.heatBtn.grid_remove()
        self.enrichBtn.grid_remove()
        self.impactLbl.grid_remove()
        self.impactBtn.grid_remove()
        self.diffsLbl.grid_remove()
        self.diffsBtn.grid_remove()
        self.costsLbl.grid_remove()
        self.costsBtn.grid_remove()
        self.yieldsLbl.grid_remove()

    # This function handles the logic behind the browse button
    def pick_file(self):
        """
        Handles the file selection and prediction process.

        1. Opens a file fialog for selecting an .xlsx file containing
        prediction data
        2. Imports the data
        3. Computes predictions
        4. Updates table with results
        5. Calculates mass balance, selectivities etc.
        Optionally saves plots

        Args:
            sheet (ctk.CTkFrame): The frame containing the table to be updated.
        """
        global default_directory, generated_file_path, file_source

        if generated_file_path is None:
            # Open a file dialog to pick a new .xlsx
            self.file_path = filedialog.askopenfilename(
            initialdir=default_directory,
            title="Select Excel File containing the conditions for prediction",
            filetypes=(("Excel files", "*.xlsx *.xls *.ods"),
                   ("All files", "*.*"))
            )
            file_source = "browsed"
        else:
            self.file_path = generated_file_path
            generated_file_path = None
            file_source = "generated"

        # Checks if file_path has been set.
        # Only relevant in case the browse window is closed before picking file
        if self.file_path:
            # Remember the folder that was picked
            default_directory = os.path.dirname(self.file_path)

            # Get file name
            self.file_name = os.path.basename(self.file_path)
            self.fileNameLbl.configure(text=f"{self.file_name}")

            global X_test, y_pred_mu, y_pred_std, sheet

            # Import data
            X_test, _ = logic.importdata(self.file_path, columns)

            # Compute predictions
            y_pred_mu, y_pred_std = predict(X_test, X_train, y_train)

            # Print predicted yields to the console
            print("Predicted Yields (average):")
            print(y_pred_mu)
            print("Predicted Yields (deviation):")
            print(y_pred_std)


            # Update table with the results
            lst_data = genTable(X_test.iloc[:, :17], y_pred_mu)
            
            # Check if the global 'sheet' exists and is not None
            if 'sheet' in globals() and sheet is not None:
                sheet.destroy()
                sheet = None  # Reset the global sheet variable

            sheet = dfTable(self, lst_data, heathers)

            # Update table with new values
            sheet.grid(row=3, column=0, columnspan=5,
                       sticky="nsew", pady=0, padx=10)

            # Calculate mass balance
            massBalance()

            # Calculate selectivities (Si) and enrichment factors (EF)
            calcStats()

            # Calculate statistical differences
            calcStatDiff()

            # Calculate the environmental impacts
            calcEnvImpacts()

            # Calculate costs
            calcCosts()

            # If the 'save plots' checkbox is checked,
            if printPlots:
                print('- Saving plots as .png')
                pltTime(save=True)
                pltTemp(save=True)
                pltSL(save=True)
                pltAcidC(save=True)
                pltPerox(save=True)
                pltAcidType(save=True)
                hmapSelect(save=True)
                pltStatDiff(save=True)
                pltEF(save=True)
                pltEI(save=True)
                pltCosts(save=True)

            self.show_plotting_buttons()

    def exportResults(self):
        """Exports all results to an Excel file."""
        try:
            print('> Exporting all results')
            
            # Create a Pandas Excel writer using openpyxl as the engine.
            output_filename = saveAsXLSX(X_test, 'all_results')
            writer = pd.ExcelWriter(os.path.join('outputs', output_filename), engine='openpyxl')

            # Convert the dataframe to openpyxl Excel object.
            X_test.to_excel(writer, sheet_name='Conditions', index=True)
            y_pred_mu.to_excel(writer, sheet_name='Predicted Yields (mu)', index=True)
            y_pred_std.to_excel(writer, sheet_name='Predicted Yields (std)', index=True)
            costs.to_excel(writer, sheet_name='Costs', index=True)
            EI.T.to_excel(writer, sheet_name='Environmental Impacts', index=True)
            EF.to_excel(writer, sheet_name='Enrichment Factors', index=True)  # Save enrichment factors
            SLi.to_excel(writer, sheet_name='Selectivity_Li', index=True)  # Save selectivity for Li
            SNi.to_excel(writer, sheet_name='Selectivity_Ni', index=True)  # Save selectivity for Ni
            SMn.to_excel(writer, sheet_name='Selectivity_Mn', index=True)  # Save selectivity for Mn
            SCo.to_excel(writer, sheet_name='Selectivity_Co', index=True)  # Save selectivity for Co
            
            # Save the Pandas Excel writer and output the Excel file.
            writer.close()

            print(f"> Save all results as {output_filename}")

        except NameError:
            print('!! Make sure predictions are run !!')


    def exportPlots(self):
        """Exports all plots as PNG images."""
        print('> Exporting all plots')
        pltTime(save=True)      
        pltTemp(save=True)
        pltSL(save=True)
        pltAcidC(save=True)
        pltPerox(save=True)
        pltAcidType(save=True)
        hmapSelect(save=True)
        pltStatDiff(save=True)
        pltEF(save=True)
        pltEI(save=True)
        pltCosts(save=True)


def validate(entry, format='float'):
    input_txt = entry.get()
    input_txt = input_txt.replace(',', '.')

    try:
        if format == 'float':
            float(input_txt)
        if format == 'int':
            int(input_txt)
        entry.configure(border_color='green')
        return True
    except ValueError:
        entry.configure(border_color='red')
        return False
    

def dfTable(parent, tableData: list, tableHeaders: list) -> Sheet:
    """
    Formats a `tksheet.Sheet` object for displaying
    conditions and yields.

    This function configures a `tksheet.Sheet` object with the provided
    data and headers, adjusts alignment and cell sizes,
    and highlights specific columns.

    Args:
        - parent: The parent widget for the sheet.
        - tableData (list): A list of lists containing the data to be
        displayed in the sheet.
        - tableHeaders (list): A list of strings representing the
        column headers.

    Returns:
        Sheet: The formatted `tksheet.Sheet` object.
    """
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
    sheet.table_align("right")
    sheet.index_align("center")
    sheet.set_all_cell_sizes_to_text()

    # Highlight results
    sheet.highlight_columns([17, 18, 19, 20], bg='#e2f2e3')
    return sheet


def predict(X_test: pd.DataFrame, X_train: pd.DataFrame,
            y_train: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts the average and standard deviation.

    This function takes test data (X_test) and uses a trained model to
    predict the average and standard deviation of extraction yields
    for Li, Ni, Mn, and Co.

    Args:
        - X_test (pd.DataFrame): The input features for prediction.
        - X_train (pd.DataFrame): The training features.
        - y_train (pd.DataFrame): The training targets.

    Returns:
        - y_pred_mu (pd.DataFrame): The predicted average extraction
        yields for each element.
        - y_pred_std (pd.DataFrame): The predicted standard deviation
        of extraction yields for each element.
        """

    print("Processing data")
    expTest_X = logic.genFeatures(X_test, X_train)
    y_test_minus_y_pred = mdl.predict(expTest_X)

    y_pred_mu, y_pred_std = logic.twinPredictorHelper(X_train, X_test, y_train,
                                                      y_test_minus_y_pred)

    cols = ['Li', 'Ni', 'Mn', 'Co']

    y_pred_mu = pd.DataFrame(data=y_pred_mu,  columns=cols)
    y_pred_std = pd.DataFrame(data=y_pred_std, columns=cols)

    return y_pred_mu, y_pred_std


def genTable(X_test: pd.DataFrame, y_pred_mu: pd.DataFrame) -> list:
    """
    Generates data for display in a table
    by combining conditions and predicted yields.

    This function takes input conditions and predicted yields and
    combines them into a single list of lists, suitable for displaying
    in a table. The data is rounded to 4 decimal places.

    Args:
        - X_test (pd.DataFrame): DF containing the input conditions.
        - y_pred_mu (pd.DataFrame): DF containing the predicted yields.

    Returns:
        - list: A list of lists representing the combined data."""

    temp = X_test.copy()
    cols_2_copy = ['Li', 'Ni', 'Mn', 'Co']

    for i, col in enumerate(cols_2_copy):
        temp[col] = y_pred_mu[col]

    # Use a tksheet to display the data
    temp = np.around(temp, 4)
    lst_data = temp.values.tolist()

    return lst_data


def scatter(title: str, xx: pd.Series, yy: pd.DataFrame,
            xlabel: str, ylabel='Leaching yield', ylim=(0, 1), save=False):
    '''
    Creates and displays/saves a scatter plot.

    The plot can be displayed directly or saved as a PNG file.

    Args:
        - title (str): The title of the plot.
        - xx (pd.Series): The x-values for the plot.
        - yy (pd.DataFrame): The y-values for the plot,
        with each column representing a different set of y-values.
        - xlabel (str): The label for the x-axis.
        - ylabel (str): The label for the y-axis.
        - ylim (tuple): The limits for the y-axis. Defaults to (0, 1).
        - save (bool): Save the plot as a PNG file. Defaults to False.
    '''

    plt.close(title)
    plt.figure(title, figsize=(6, 4))

    lb = ['Li', 'Ni', 'Mn', 'Co']
    mk = ['o', 's', '^', 'd']

    for i, col in enumerate(lb):
        plt.scatter(xx, yy[col], label=lb[i], marker=mk[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend(loc='best')

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'

        os.makedirs('outputs/figures', exist_ok=True)

        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def pltTime(save=False):
    '''
    Plot time vs yield.
    '''
    if not save:
        print('* PLOT: yield vs time scatter')

    xx = X_test['time']
    try:
        yy = y_pred_mu

        scatter(title='Time',
                xx=xx, yy=yy,
                xlabel='Time (min)', save=save)
    except NameError:
        print('!! Make sure predictions are run !!')

def pltTemp(save=False):
    '''
    Plot temperature vs yield
    '''
    if not save:
        print('* PLOT: yield vs temperature scatter')

    xx = X_test['temp']

    try:
        yy = y_pred_mu

        scatter(title='Temperature',
                xx=xx, yy=yy,
                xlabel='Temperature (°C)', save=save)
    except NameError:
        print('!! Make sure predictions are run !!')


def pltSL(save=False):
    '''
    Plot solid:liquid ratio vs yield.
    '''
    if not save:
        print('* PLOT: yield vs S/L scatter')

    xx = X_test['solidToLiquid']
    try:
        yy = y_pred_mu

        scatter('S-L ratio', xx, yy, 'S/L (g/L)', save=save)
    except NameError:
        print('!! Make sure predictions are run !!')


def pltAcidC(save=False):
    '''
    Plot acid concentration vs yield.
    '''
    if not save:
        print('* PLOT: yield vs [acid] scatter')

    xx = X_test['acidC']

    try:
        yy = y_pred_mu
        scatter('Acid concentration', xx, yy, '[acid] (mol/L)', save=save)
    except NameError:
        print('!! Make sure predictions are run !!')


def pltPerox(save=False):
    '''
    Plot peroxide concentration vs yield.
    '''
    if not save:
        print('* PLOT: yield vs [H2O2] scatter')

    xx = X_test['H2O2_conc']
    try:
        yy = y_pred_mu

        scatter('H2O2 concentration', xx, yy, '[H2O2] (mol/L)', save=save)
    except NameError:
        print('!! Make sure predictions are run !!')


def pltAcidType(save=False):
    '''
    Plot yield by acid type.
    '''
    if not save:
        print('* PLOT: yield vs acid type bar chart')

    title = 'Acid type'

    plt.close(title)
    plt.figure(title, figsize=(6, 4))

    acid = X_test['pKa1']

    try:
        combined_df = pd.concat([acid, y_pred_mu], axis=1)
    except NameError:
        print('!! Make sure predictions are run !!')
        plt.close(title)
        return

    # Melt the DataFrame to long format for seaborn
    df_melted = pd.melt(combined_df, id_vars='pKa1',
                        var_name='Metal', value_name='Yield')

    df_melted['Acid'] = df_melted['pKa1'].map(pka_map)

    sns.barplot(data=df_melted, x='Acid', y='Yield', hue='Metal')

    plt.ylabel('Leaching Yield')
    plt.title('Leaching Yields with Different Acids')

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def calcStats():
    '''
    Calculates the selectivities and enrichment factors for the
    predicted yields.

    This function calculates the selectivities of Li, Ni, Mn, and Co
    relative to each other based on the predicted yields (`y_pred_mu`).
    It also calculates the enrichment factors (EF) by comparing the
    molar ratios of these elements in the leachate to their initial
    molar ratios in the NMC material.

    The function uses global variables:
        - `SLi`, `SNi`, `SMn`, `SCo`: DataFrames storing the selectivities.
        - `EF`: DataFrame storing the enrichment factors.
    '''

    global SLi, SNi, SMn, SCo, EF

    # Calculate the selectivities
    SLi = pd.DataFrame()
    SNi = pd.DataFrame()
    SMn = pd.DataFrame()
    SCo = pd.DataFrame()

    lb = ['Li', 'Ni', 'Mn', 'Co']

    for i in lb:
        SLi[i] = y_pred_mu[i]/y_pred_mu['Li']
        SNi[i] = y_pred_mu[i]/y_pred_mu['Ni']
        SMn[i] = y_pred_mu[i]/y_pred_mu['Mn']
        SCo[i] = y_pred_mu[i]/y_pred_mu['Co']

    # Calculate enrichment factors (EF)

    # Type of NMC
    # Li_l Ni_n Mn_m Co_c
    ni = X_test['inputNi'].to_numpy()
    mn = X_test['inputMn'].to_numpy()
    co = X_test['inputCo'].to_numpy()
    li = np.ones(ni.size)

    # Each line corresponds to a condition
    # Each column is the ratio of each metal to Li
    NMCType = np.transpose(np.vstack((li, ni, mn, co)))

    # Calculate the "molar ratio" of each metal compared to every other metal
    NMC_Composition = NMCType/NMCType.sum(axis=1, keepdims=True)

    # Calculate the metal in the leachate using the predicted yields
    metal_in_leachate = NMCType * y_pred_mu.to_numpy()

    # Calculate the "molar ratio" in the leachate
    leachCompo = metal_in_leachate/metal_in_leachate.sum(axis=1, keepdims=True)

    # Compare how "rich" each metal is in the leachate vs in the initial matrix
    EF = leachCompo/NMC_Composition

    # Turn the EF numpy matrix into a dataframe
    # for consistency and easier handling
    cols = ['Li', 'Ni', 'Mn', 'Co']

    EF = pd.DataFrame(data=EF,  columns=cols)


def hmapSelect(save=False):
    '''
    Plot selectivities as heatmaps.
    '''
    if not save:
        print('* PLOT: Heatmap of selectivities')

    # Display heatmaps
    title = 'Selectivity Heatmap'
    plt.close(title)
    plt.figure(title)

    def hmap(index: int, df: pd.DataFrame, title: str, color: str):
        df.index += 1  # To match the table in the GUI
        plt.subplot(2, 2, index)
        sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cmap=color)
        plt.xlabel('Metals')
        plt.ylabel('Conditions')
        plt.title('Selectivity vs ' + title)
        plt.tight_layout()

    try:
        hmap(1, SLi, 'Li', 'Blues')
        hmap(2, SNi, 'Ni', 'Oranges')
        hmap(3, SMn, 'Mn', 'Greens')
        hmap(4, SCo, 'Co', 'Reds')
    except NameError:
        print('!! Make sure predictions are run !!')
        plt.close(title)
        return

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()

def pltEF(save=False):
    '''
    Plot enrichment factors as heatmaps.
    '''
    if not save:
        print('* PLOT: Enrichment factor bar chart')

    # Display heatmaps
    title = 'EF Heatmap'
    plt.close(title)
    plt.figure(title, figsize=(6, 4))

    try:
        EF.index += 1
    except NameError:
        print('!! Make sure predictions are run !!')
        plt.close(title)
        return

    sns.heatmap(EF, annot=True, fmt=".2f", linewidths=.5, cmap='BuPu')
    plt.xlabel('Metals')
    plt.ylabel('Conditions')
    plt.title('Enrichment Factor (EF)')
    plt.tight_layout()

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def pairwise_t_tests(means_df, stds_df, n, col='Li', alpha=0.05):
    """
    Performs pairwise t-tests with Bonferroni correction,
    taking means and standard deviations from separate DataFrames
    and using a fixed sample size.

    Args:
        - means_df: DataFrame containing the means.
        - stds_df: DataFrame containing the standard deviations.
        - n: Sample size (integer).
        - col: Name of the column with the means and std.
        - alpha: Significance level.

    Returns:
        A pandas DataFrame with the results of the pairwise t-tests.
        Significant=False means that there is not enough evidence
        to conclude that the two tests are significantly different.
    """
    print('- Computing pairwise t-tests for ', col)

    if len(means_df) != len(stds_df):
        errortxt = "Means and standard deviations DataFrames\
                    must have the same length."
        raise ValueError(errortxt)

    k = len(means_df)
    num_comparisons = k * (k - 1) / 2
    bonferroni_alpha = alpha / num_comparisons
    results = []

    for (i, row1), (j, row2) in combinations(means_df.iterrows(), 2):
        t_stat, p_value = stats.ttest_ind_from_stats(
            row1[col], stds_df.loc[i, col], n,
            row2[col], stds_df.loc[j, col], n,
            equal_var=False)
        results.append({
            "group1": i + 1,
            "group2": j + 1,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < bonferroni_alpha})

    return pd.DataFrame(results)


def calcStatDiff():
    '''
    Calculates pairwise t-tests and saves them in global variables.
    '''
    global compLi, compNi, compMn, compCo

    # Perform pairwise t-tests
    n = y_train.shape[0]

    compLi = pairwise_t_tests(y_pred_mu, y_pred_std, n, 'Li')
    compNi = pairwise_t_tests(y_pred_mu, y_pred_std, n, 'Ni')
    compMn = pairwise_t_tests(y_pred_mu, y_pred_std, n, 'Mn')
    compCo = pairwise_t_tests(y_pred_mu, y_pred_std, n, 'Co')

    if printResults:
        fname = saveAsXLSX(compLi, 'ttest_Li')
        print('> Save t-tests as ', fname)

        fname = saveAsXLSX(compNi, 'ttest_Ni')
        print('> Save t-tests as ', fname)

        fname = saveAsXLSX(compMn, 'ttest_Mn')
        print('> Save t-tests as ', fname)

        fname = saveAsXLSX(compCo, 'ttest_Co')
        print('> Save t-tests as ', fname)


def pltStatDiff(save=False):
    '''
    Plots statistical differences.
    '''
    if not save:
        print('* PLOT: Statistical differences')
    # Display heatmaps
    title = 'Comparisons Heatmap'
    plt.close(title)
    plt.figure(title)
    plt.suptitle('Highlighted cells: not sig. different (p > 0.05)',
                 fontsize=12)

    def hmap(index: int, df: pd.DataFrame, title: str, color: str):
        # Reformat the df for the heatmap
        df = df.pivot(index='group1', columns='group2', values='significant')

        df = df.fillna(True)
        df = ~df

        # Setup the plots themselves
        plt.subplot(2, 2, index)
        sns.heatmap(df, annot=False, cbar=False, linewidths=.5,
                    linecolor='lightgrey', cmap=color)
        plt.xlabel('Condition 1')
        plt.ylabel('Condition 2')
        plt.title(title)
        plt.tight_layout()
    
    try:
        hmap(1, compLi, 'Li', 'Blues')
        hmap(2, compNi, 'Ni', 'Oranges')
        hmap(3, compMn, 'Mn', 'Greens')
        hmap(4, compCo, 'Co', 'Reds')
    except NameError:
        print('!! Make sure predictions are run !!')
        plt.close(title)
        return

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def massBalance():
    '''
    Calculates the mass of concentrated acid and volume of water needed
    for the leaching process, accounting for acid purity.

    The function uses global variables:
        - `X_test`: DataFrame containing the reaction conditions.
        - `MB`: DataFrame to store the calculated mass balance data.
        - `printResults`: Boolean indicating whether to save the results
        to an Excel file.
    '''

    global MB
    mLIB = 1000  # kg of LIB cathode

    print('- Computing mass balance')

    # Create a df with only the data that's needed for the env. impacts
    MB = pd.DataFrame()
    cols_2_copy = ['pKa1', 'acidC', 'solidToLiquid', 'temp', 'time']

    for i, col in enumerate(cols_2_copy):
        MB[col] = X_test[col]

    MB = MB.rename(columns={'solidToLiquid': 'SL'})
    MB.insert(0, 'Acid', MB['pKa1'].map(pka_map))

    # Mass of acid used - acidC, S/L -> mDilAcid, mAcid
    MB['vDilAcid'] = mLIB * 1e3 / MB['SL']  # Liters of dilute acid used
    MB['molAcid'] = MB['vDilAcid'] * MB['acidC']  # mol of acid

    # Get Mr from acid props
    Mr_map = acidProps.set_index('Acid')['Mr'].to_dict()
    MB['Mr'] = MB['Acid'].map(Mr_map)

    # Get purity from acid props
    purity_map = acidProps.set_index('Acid')['Purity'].to_dict()
    MB['Purity'] = MB['Acid'].map(purity_map)

    # Mass of pure acid, kg
    MB['mAcid_pure'] = MB['Mr'] * MB['molAcid'] * 1e-3

    # Correct for purity: mass of actual acid solution needed
    MB['mAcid'] = MB['mAcid_pure'] / MB['Purity']

    # Volume of pure acid - m(Acid) * density (pure acid)
    dens_map = acidProps.set_index('Acid')['density'].to_dict()
    MB['vAcid'] = MB['mAcid'] / MB['Acid'].map(dens_map) # density is mass/volume, so volume = mass/density

    # Volume of water - V(dilute acid) -  V(pure acid)
    MB['vH2O'] = MB['vDilAcid']-MB['vAcid']

    if printResults:
        fname = saveAsXLSX(MB, 'MB')
        print('> Save mass balance as ', fname)


def calcEnvImpacts():
    '''
    Estimates the environmental impacts of the leaching process.

    This function calculates the env. impacts based on the mass balance
    results (`MB`) and the env. impact multipliers for each acid
    (`acid_EnvImpacts`). It multiplies the mass of each acid used by
    its env. impact multipliers to obtain the overall impact.

    The function uses global variables:
        - `MB`: DataFrame containing the mass balance data.
        - `acid_EnvImpacts`: DataFrame containing the env. impact
        multipliers for each acid.
        - `EI`: DataFrame to store the calculated env. impacts.
        - `printResults`: Boolean indicating whether to save
        the results to an Excel file.
    '''
    global EI, rankedEI

    print('- Computing environmental impacts')
    # Import and format the acid impact multipliers to consider
    auxImpacts = acid_EnvImpacts.T
    auxImpacts = auxImpacts.drop(auxImpacts.index[0])
    auxImpacts = auxImpacts.reset_index()
    auxImpacts = auxImpacts.rename_axis(None, axis=1)
    auxImpacts = auxImpacts.rename(columns={'index': 'Acid'})

    # Create a temporary dataframe with everything needed for the calculation
    temp = pd.merge(MB, auxImpacts, on='Acid', how='left', copy=True)

    # Number of columns that were added
    N = auxImpacts.shape[1]-1
    multiplier = temp['mAcid']
    # Select the N rightmost columns
    cols_to_multiply = temp.columns[-N:]

    # Multiply the selected columns by the multiplier
    temp[cols_to_multiply] = temp[cols_to_multiply].mul(multiplier, axis=0)
    temp[cols_to_multiply] = temp[cols_to_multiply].astype(float)

    # Copy the relevant columns from the temp DF into a new one
    # to store the environmental impacts, EI
    EI = temp[cols_to_multiply].copy()

    EI.index +=1

    # Remove rows with NaN values from EI
    EI = EI.dropna()

    # Rank the environmental impacts
    rankedEI = EI.rank(axis=0, method='first')

    # Calculate the score for each condition
    rankedEI['Score'] = rankedEI.sum(axis=1)

    # Rank the overall score
    rankedEI['Overall Rank'] = rankedEI['Score'].rank(method='min').astype(int)

    if printResults:
        fname = saveAsXLSX(EI, 'EI')
        print('> Save impacts as ', fname)


def calcCosts():
    '''
    Calculates the costs of reagents and heating for each
    leaching condition.

    It uses the mass balance results (`MB`) and the cost data for each
    acid (`acid_Costs`) to estimate the total cost for each set of conditions.

    The function uses global variables:
        - `MB`: DataFrame containing the mass balance data.
        - `acid_Costs`: DataFrame containing the cost data for each acid.
        - `costs`: DataFrame to store the calculated costs.
        - `printResults`: Boolean indicating whether to save the
        results to an Excel file.
    '''
    print('- Computing costs')

    global costs

    costs = pd.DataFrame()
    temp = pd.DataFrame()
    mLIB = 1000  # kg of LIB cathode
    initial_temp = 25  # Initial temperature in Celsius

    # Get price per kg of the acids
    acidPrice_map = acidProps.set_index('Acid')['price kg'].to_dict()
    temp['acid price'] = MB['Acid'].map(acidPrice_map)

    # Calculate the cost of concentrated acid
    costs['Acid Cost, €'] = MB['mAcid'] * temp['acid price']

    # Get NMC density
    temp['NMC volume'] = mLIB / 2.11  # Liters of NMC

    # Total volume = Vliquid + VNMC in m3
    temp['Total m3'] = (temp['NMC volume'] + MB['vDilAcid']) * 1e-3

    # Get mixing power needs
    mixing_map = acidProps.set_index('Acid')['mixing kwm3'].to_dict()
    temp['mix kwm3'] = MB['Acid'].map(mixing_map)

    # Calculate mixing cost
    costs['Mixing, kWh'] = temp['mix kwm3'] * temp['Total m3'] * MB['time'] / 60
    costs['Mixing, €'] = costs['Mixing, kWh'] * 0.1899

    # Heating cost calculation
    water_density = 1  # kg/L
    water_heat_capacity = 4.186  # kJ/kg/°C

    # Calculate the mass of water in kg
    mass_water = MB['vH2O'] * water_density

    # Calculate the energy needed to heat the water
    delta_T = abs(MB['temp'] - initial_temp)
    energy_needed_kJ = mass_water * water_heat_capacity * delta_T

    # Convert energy from kJ to kWh
    energy_needed_kWh = energy_needed_kJ / 3600

    # Calculate heating cost
    costs['Heating, kWh'] = energy_needed_kWh
    costs['Heating, €'] = costs['Heating, kWh'] * 0.1899

    # Normalize costs by the total amount of metal leached
    molar_masses = {'Li': 6.94, 'Ni': 58.69, 'Mn': 54.94, 'Co': 58.93}
    total_metal_leached = (y_pred_mu * molar_masses).sum(axis=1) / 1000  # kg of metal leached

    costs['Acid Cost, €/kg'] = costs['Acid Cost, €'] / total_metal_leached
    costs['Mixing, €/kg'] = costs['Mixing, €'] / total_metal_leached
    costs['Heating, €/kg'] = costs['Heating, €'] / total_metal_leached

    # Fix indexes so that they match the other tables
    costs.index += 1

    if printResults:
        fname = saveAsXLSX(costs, 'costs')
        print('> Save costs as ', fname)


def pltEI(save=False):
    '''
    Plot the ranked environmental impacts.
    '''
    global file_source, rankedEI

    # Plotting the ranked environmental impacts as a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(rankedEI.drop(['Score'], axis=1).T, annot=True, fmt=".0f", linewidths=.5, cmap="RdYlGn_r")
    plt.title('Ranked Environmental Impacts (Lower is better)')
    plt.ylabel('Environmental Impact Category')

    if file_source == "generated":
        column_name = column_mapping.get(variable_to_vary, variable_to_vary)
        plt.xlabel(variable_to_vary)
        rounded_labels = X_test[column_name].round(1)
        plt.xticks(ticks=np.arange(len(X_test[column_name]))+0.5, labels=rounded_labels)
    else:
        plt.xlabel('Conditions')
        plt.xticks(ticks=np.arange(len(rankedEI)) + 0.5, labels=rankedEI.index)

    plt.tight_layout()

    title = 'Environmental_Impacts'

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def pltCosts(save=False):
    '''
    Plot the costs of reagents and heating per kg of metal leached.
    '''
    if not save:
        print('* PLOT: Cost bar chart')

    title = 'Costs'

    plt.close(title)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), num=title)

    try:
        costs['Conditions'] = costs.index
    except NameError:
        print('!! Make sure predictions are run !!')
        plt.close(title)
        return

    # Plot acid costs per kg leached
    sns.barplot(x='Conditions', y='Acid Cost, €/kg', data=costs, ax=axes[0])
    axes[0].set_title('Acid cost per kg leached')

    sns.barplot(x='Conditions', y='Mixing, €/kg', data=costs, ax=axes[1])
    axes[1].set_title('Mixing cost per kg leached')

    sns.barplot(x='Conditions', y='Heating, €/kg', data=costs, ax=axes[2])
    axes[2].set_title('Heating cost per kg leached')

    plt.tight_layout()

    if save:
        curr_datetime = datetime.datetime.now()
        fname = title + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.png'
        print('> Save plot as ', fname)
        plt.savefig('outputs/figures/' + fname, bbox_inches='tight')
        plt.close(title)
    else:
        plt.show()


def saveAsXLSX(df: pd.DataFrame, fname: str, path='outputs/') -> str:
    '''
    Saves a DataFrame to an Excel file with a timestamped filename.

    Args:
        - df (pd.DataFrame): The DataFrame to be saved.
        - fname (str): The base filename (without extension) to be used.
        - path (str): The directory path where the file will be saved.
        Defaults to 'outputs/'.

    Returns:
        str: Complete filename (+ timestamp and extension) of the saved file.
    '''
    curr_datetime = datetime.datetime.now()
    fname = fname + '_' + curr_datetime.strftime("%Y%m%d_%H%M%S") + '.xlsx'

    os.makedirs('outputs', exist_ok=True)

    df.to_excel(path + fname)
    return fname


def getPath(relPath: str) -> str:
    '''
    Builds the full absolute path from a relative path.
    '''
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relPath))


# Flags to set whether results are saved
printResults = False
printPlots = False

generated_file_path = None
variable_to_vary = None
file_source = None

# Set the default directory
default_directory = os.getcwd()

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# Load pre-trained PD-GBR model
path = getPath('model/PD-GBR_full.gz')
mdl = joblib.load(path)


# Load training data
path = getPath('model/xtrain_full.gz')
X_train = joblib.load(path)

path = getPath('model/ytrain_full.gz')
y_train = joblib.load(path)


columns = ['inputNi', 'inputMn', 'inputCo',
           'temp', 'pKa1', 'pKa2', 'pKa3',
           'nProtons', 'SMILES', 'sLi', 'sNi', 'sMn', 'sCo',
           'acidC', 'H2O2_conc', 'solidToLiquid','time']

# Import sample data
path = getPath('data/template.xlsx')
X_test, _ = logic.importdata(path, cols=columns)

heathers = ['inputNi', 'inputMn', 'inputCo', 'T (°C)', 'pKa1', 'pKa2',
            'pKa3', 'nProtons', 'SMILES', 'sLi (g/100ml)', 'sNi (g/100ml)',
            'sMn (g/100ml)', 'sCo (g/100ml)', '[acid] (M)',
            '[H2O2]', 'S/L (g/L)', 't (min)',
            'Li', 'Ni', 'Mn', 'Co']

column_mapping = {
    'T (°C)': 'temp',
    '[acid] (M)': 'acidC',
    '[H2O2]': 'H2O2_conc',
    'S/L (g/L)': 'solidToLiquid',
    't (min)': 'time'
}



# Import acid properties from excel sheet
acidProps = pd.read_excel(getPath('data/Acid_properties.xlsx'), header=0)

pka_map = acidProps.set_index('pKa1')['Acid'].to_dict()

# Load the impacts table
acid_EnvImpacts = pd.read_excel(getPath('data/Acids_impacts.xlsx'),
                                header=0, index_col=0)
acid_Costs = pd.read_excel(getPath('data/Acids_costs.xlsx'),
                           header=0, index_col=0)

matplotlib.use('TkAgg')

# Filter out a FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

app = App()
app.mainloop()
