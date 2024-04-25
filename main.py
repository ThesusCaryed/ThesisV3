from tkinter import *
from PIL import Image, ImageTk
import customtkinter
import subprocess
from developer_functionality import developer_functionality  # Import the function

# Set appearance mode and default color theme
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

def on_image_recognizer_click():
    print("Image Recognizer clicked")

def on_results_click():
    print("Results button clicked")

def on_developer_click():
    developer_functionality(root)

def on_exit_click():
    root.quit()

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition")
        
        # Set background color to beige
        self.root.configure(bg="#9DC3E2")  # Use hexadecimal color code for beige

        bg_image = Image.open(r"./assets/bg1.jpg")
        bg_image = bg_image.resize((900, 480), Image.LANCZOS)
        self.photoimg_bg = ImageTk.PhotoImage(bg_image)
        self.canvas = Canvas(root, width=900, height=480, bg="#242424", bd=0, highlightthickness=0)  # Set canvas background color to transparent
        # Calculate the center position of the canvas
        center_x = (900 - bg_image.width) // 2  # Remove parentheses from bg_image.width
        center_y = (480 - bg_image.height) // 2  # Remove parentheses from bg_image.height
        x = -10  # Adjust the x-coordinate as needed
        y = 100   # Adjust the y-coordinate as needed
        self.canvas.create_image(x, y, anchor=NW, image=self.photoimg_bg)
        self.canvas.pack(side=TOP)

        # Default position for HumId label
        self.humid_position = (230, 45)

        # Add "HumId" label at the top-left
        self.label = customtkinter.CTkLabel(self.root, text="HumId: Human Identification", font=("Poppins Bold", 30))
        self.label.pack(pady=10, padx=10, anchor="nw")  # Increase padx for left alignment
        self.set_humid_position(self.humid_position)

        # Load images
        img_recognizer = Image.open(r"assets\facerecognizer.jpg")
        img_recognizer = img_recognizer.resize((150, 150), Image.LANCZOS)
        self.photoimg_recognizer = ImageTk.PhotoImage(img_recognizer)

        img_results = Image.open(r"assets\result.jpg")
        img_results = img_results.resize((150, 150), Image.LANCZOS)
        self.photoimg_results = ImageTk.PhotoImage(img_results)

        img_developer = Image.open(r"assets\developer.jpg")
        img_developer = img_developer.resize((150, 150), Image.LANCZOS)
        self.photoimg_developer = ImageTk.PhotoImage(img_developer)

        img_exit = Image.open(r"assets\exit.jpg")
        img_exit = img_exit.resize((150, 150), Image.LANCZOS)
        self.photoimg_exit = ImageTk.PhotoImage(img_exit)

        # Font configuration
        button_font = ("Poppins Medium", 12)

        # Create transparent image labels as buttons
        self.label_recognizer = Label(self.root, image=self.photoimg_recognizer, cursor="hand2")
        self.label_recognizer.place(x=300, y=200)
        self.label_recognizer.bind("<Button-1>", lambda event: on_image_recognizer_click())
        self.label_recognizer.bind("<Enter>", lambda event: self.on_enter(self.label_recognizer, "#385B88"))  # Change color on enter
        self.label_recognizer.bind("<Leave>", lambda event: self.on_leave(self.label_recognizer, "#1f2833"))  # Change color on leave

        self.button_recognizer = Button(self.root, text="Image Recognizer", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_image_recognizer_click, width=15, height=1, font=button_font)
        self.button_recognizer.place(x=300, y=360)
        self.button_recognizer.bind("<Enter>", lambda event: self.on_enter(self.button_recognizer, "#385B88"))  # Change color on enter
        self.button_recognizer.bind("<Leave>", lambda event: self.on_leave(self.button_recognizer, "#1f2833"))  # Change color on leave

        self.label_results = Label(self.root, image=self.photoimg_results, cursor="hand2")
        self.label_results.place(x=500, y=200)
        self.label_results.bind("<Button-1>", lambda event: on_results_click())
        self.label_results.bind("<Enter>", lambda event: self.on_enter(self.label_results, "#385B88"))  # Change color on enter
        self.label_results.bind("<Leave>", lambda event: self.on_leave(self.label_results, "#1f2833"))  # Change color on leave

        self.button_results = Button(self.root, text="Saved Images", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_results_click, width=15, height=1, font=button_font)
        self.button_results.place(x=500, y=360)
        self.button_results.bind("<Enter>", lambda event: self.on_enter(self.button_results, "#385B88"))  # Change color on enter
        self.button_results.bind("<Leave>", lambda event: self.on_leave(self.button_results, "#1f2833"))  # Change color on leave

        self.label_developer = Label(self.root, image=self.photoimg_developer, cursor="hand2")
        self.label_developer.place(x=700, y=200)
        self.label_developer.bind("<Button-1>", lambda event: on_developer_click())
        self.label_developer.bind("<Enter>", lambda event: self.on_enter(self.label_developer, "#385B88"))  # Change color on enter
        self.label_developer.bind("<Leave>", lambda event: self.on_leave(self.label_developer, "#1f2833"))  # Change color on leave

        self.button_developer = Button(self.root, text="Developer", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_developer_click, width=15, height=1, font=button_font)
        self.button_developer.place(x=700, y=360)
        self.button_developer.bind("<Enter>", lambda event: self.on_enter(self.button_developer, "#385B88"))  # Change color on enter
        self.button_developer.bind("<Leave>", lambda event: self.on_leave(self.button_developer, "#1f2833"))  # Change color on leave

        self.label_exit = Label(self.root, image=self.photoimg_exit, cursor="hand2")
        self.label_exit.place(x=900, y=200)
        self.label_exit.bind("<Button-1>", lambda event: on_exit_click())
        self.label_exit.bind("<Enter>", lambda event: self.on_enter(self.label_exit, "#385B88"))  # Change color on enter
        self.label_exit.bind("<Leave>", lambda event: self.on_leave(self.label_exit, "#1f2833"))  # Change color on leave

        self.button_exit = Button(self.root, text="Exit", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_exit_click, width=15, height=1, font=button_font)
        self.button_exit.place(x=900, y=360)
        self.button_exit.bind("<Enter>", lambda event: self.on_enter(self.button_exit, "#385B88"))  # Change color on enter
        self.button_exit.bind("<Leave>", lambda event: self.on_leave(self.button_exit, "#1f2833"))  # Change color on leave

    def set_humid_position(self, position):
        # Set the position of the "HumId" label
        self.label.place(x=position[0], y=position[1], anchor="nw")

    def on_enter(self, widget, color):
        widget.config(bg=color)

    def on_leave(self, widget, color):
        widget.config(bg=color)

if __name__ == "__main__":
    root = customtkinter.CTk()
    obj = Face_Recognition(root)
    root.mainloop()