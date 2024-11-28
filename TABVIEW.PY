import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

class MyTabView(customtkinter.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # create tabs
        self.add("tab 1")
        self.add("tab 2")

        # add widgets on tabs
        # TAB 1
        self.label = customtkinter.CTkLabel(master=self.tab("tab 1"), text="This is tab 1")
        self.label.grid(row=0, column=0, padx=20, pady=10)

        # TAB 2
        self.label = customtkinter.CTkLabel(master=self.tab("tab 2"), text="This is tab 2")
        self.test1 = customtkinter.CTkButton(master=self.tab("tab 2"), 
                                text="Frame 1", 
                                anchor="center",
                                fg_color="transparent", 
                                border_color="#3B8ED0", 
                                border_width=2,  
                                command=self.one_event)
        
        self.label.grid(row=0, column=0, padx=20, pady=10)
        self.test1.grid(row=1, column=0, padx=20, pady=10)
        
    def one_event(self):
        print("Frame 1 button clicked")





class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.tab_view = MyTabView(master=self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20)


app = App()
app.mainloop()