import customtkinter as ctk
import pandastable as ptable
import logic
import os

from tkinter import filedialog

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

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
        print("Options frame open")
        self.controller.optFrame.tkraise()
    def inputsEvent(self):
        print("Inputs frame open")
        self.controller.inputsFrame.tkraise()
    def predEvent(self):
        print("Predictions frame open")
        self.controller.predsFrame.tkraise()
    def impactEvent(self):
        print("Impacts frame open")
        self.controller.impactFrame.tkraise()
    def summaryEvent(self):
        print("Summary frame open")
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
        # self.grid_columnconfigure(1, weight=0, uniform="group1")
        self.grid_rowconfigure((0,2), weight=0)
        self.grid_rowconfigure((1, 3, 4),weight=1)


        # Title of the window with some explanation
        title = ctk.CTkLabel(self, 
                            text="These are the conditions for which the yields will be predicted")
        title.grid(row=0, column=0, columnspan=2, pady=(10,0))

        # Label and button to allow picking of a test data file
        self.file_path = ""
        pick_file_button = ctk.CTkButton(
            master=self, 
            text="Browse...", 
            command=self.pick_file)
        pick_file_button.grid(row=1, column=1, sticky="w", padx=(5,5), pady=(10,))

        self.fileLabel = ctk.CTkLabel(self,
                                      text="No file selected.",
                                      font=("Helvetica", 14, 'bold'))
        self.fileLabel.grid(row=1, column=0, sticky="e", padx=(5,5), pady=(10,))

        # Display the table using a simple textbox
        # A simple label for the table
        tableLabel=ctk.CTkLabel(self,
                                text="Conditions:",
                                font=("Helvetica", 14, 'bold'))
        tableLabel.grid(row=2, column=0, columnspan=2, sticky="sw", padx=10)
        # The textbox the table is displayed in
        self.textbox = ctk.CTkTextbox(self)
        self.textbox.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(0,20), padx=(20,20))

    # This function handles the logic behind the browse button
    def pick_file(self):
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

            X_test,_ = logic.importdata(self.file_path)
            # print(X_test)

            # Convert DataFrame to string
            table_str = X_test.to_string(index=False)  # Exclude index if not needed

            self.textbox.insert("0.0", table_str)  # Insert at the beginning
            self.textbox.configure(state="disabled")  # Make it read-only

            # Try using a pandastable


class FramePreds(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self,
                             text="ML RESULTS HERE")
        title.grid(row=0, column=0)

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

app = App()
app.mainloop()