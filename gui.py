import customtkinter

customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("900x600")
        self.title("LIB Recycling Helper")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.navigation = NavigationFrame(self, controller=self)
        self.navigation.grid(row=0, column=0, sticky="nesw")

        self.frame5 = Frame5(self, controller=self)
        self.frame4 = Frame4(self, controller=self)
        self.frame3 = Frame3(self, controller=self)
        self.frame2 = Frame2(self, controller=self)
        self.frame1 = Frame1(self, controller=self)

        self.frame1.grid(row=0, column=1, sticky="nsew")
        self.frame2.grid(row=0, column=1, sticky="nsew")
        self.frame3.grid(row=0, column=1, sticky="nsew")
        self.frame4.grid(row=0, column=1, sticky="nsew")
        self.frame5.grid(row=0, column=1, sticky="nsew")

# Setup navigation bar on the lefthand side of the window
class NavigationFrame(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(fg_color="grey76")
        self.configure(border_width=2)

        # Title
        title = customtkinter.CTkLabel(self, 
                                       text="LIB RECYCLING",
                                       anchor="center",
                                       font=("Helvetica", 18, 'bold'))
        title.grid(row=0, column=0, padx=(20, 20),pady=(10,0))

        # Add buttons to access the different steps
        btnOptions = customtkinter.CTkButton(self, 
                                        text="0. Options",
                                        anchor="center",
                                        command=self.optnEvent,
                                        font=("Helvetica", 14))
        btnOptions.grid(row=1, column=0, padx=(20, 20), pady=(10,10))

        btnInputs = customtkinter.CTkButton(self, 
                                        text="1. Inputs",
                                        anchor="center",
                                        command=self.inputsEvent,
                                        font=("Helvetica", 14))
        btnInputs.grid(row=2, column=0, padx=(20, 20), pady=(10,10))

        btnPred = customtkinter.CTkButton(self, 
                                        text="2. Predictions",
                                        anchor="center",
                                        command=self.predEvent,
                                        font=("Helvetica", 14))
        btnPred.grid(row=3, column=0, padx=(20, 20), pady=(10,10))

        btnImpact = customtkinter.CTkButton(self, 
                                        text="3. Eco analysis",
                                        anchor="center",
                                        command=self.impactEvent,
                                        font=("Helvetica", 14))
        btnImpact.grid(row=4, column=0, padx=(20, 20), pady=(10,10))

        btnSummary = customtkinter.CTkButton(self, 
                                        text="4. Summary",
                                        anchor="center",
                                        command=self.summaryEvent,
                                        font=("Helvetica", 14))
        btnSummary.grid(row=5, column=0, padx=(20, 20), pady=(10,10))

    # These fucntions are called when the buttons above are pressed
    def optnEvent(self):
        print("Options button clicked")
        self.controller.frame1.tkraise()
    def inputsEvent(self):
        print("Inputs button clicked")
        self.controller.frame2.tkraise()
    def predEvent(self):
        print("Predictions button clicked")
        self.controller.frame3.tkraise()
    def impactEvent(self):
        print("Impacts button clicked")
        self.controller.frame4.tkraise()
    def summaryEvent(self):
        print("Summary button clicked")
        self.controller.frame5.tkraise()

# Setup the different windows that appear in the right side of the window
class Frame1(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="OPTIONS HERE", 
                                       anchor="center")
        title.grid(row=0, column=0)

class Frame2(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="INPUTS HERE", 
                                       anchor="center")
        title.grid(row=0, column=0)

class Frame3(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="ML RESULTS HERE", 
                                       anchor="center")
        title.grid(row=0, column=0)

class Frame4(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="ECO RESULTS HERE", 
                                       anchor="center")
        title.grid(row=0, column=0)

class Frame5(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="SUMMARY/EXPORT HERE", 
                                       anchor="center")
        title.grid(row=0, column=0)
              
app = App()
app.mainloop()