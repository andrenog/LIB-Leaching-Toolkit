import customtkinter

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")

# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Window/App Customization
        self.geometry("500x250")
        self.title("Starter Code Example")

        # Grid Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Widgets
        label = customtkinter.CTkLabel(self, 
                                       text="Hello Blog World!",
                                       font=("Arial", 25),
                                       anchor="center")
        button = customtkinter.CTkButton(self, text="Click Me!")

        # Placing Widgets on Grid
        label.grid(row=0, column=0)
        button.grid(row=1, column=0)

app = App()
app.mainloop()