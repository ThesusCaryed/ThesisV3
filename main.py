from tkinter import *
from PIL import Image, ImageTk
import customtkinter
import subprocess
import tkinter as tk
from developer_functionality import developer_functionality  # Import the function
from results_display import ResultsWindow
from testmodel import main as run_face_recognition
import threading

# Set appearance mode and default color theme
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

def on_image_recognizer_click():
    print("Image Recognizer clicked")
    thread = threading.Thread(target=run_face_recognition)
    thread.start()

def on_results_click():
    print("Results button clicked")
    results_root = tk.Toplevel()  # Creates a new top-level window
    results_root.title("Results Display")
    results_app = ResultsWindow(results_root)  # Initializes the results window
    results_root.mainloop()  # Ensures the new window's event loop runs
    print("Results window should now be open.")

def on_developer_click():
    developer_functionality(root)

def on_exit_click():
    root.quit()

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.title("Face Recognition")
        
        self.root.configure(bg="#9DC3E2")

        bg_image = Image.open(r"assets\aaa.jpg")
        bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
        self.photoimg_bg = ImageTk.PhotoImage(bg_image)
        self.canvas = Canvas(root, width=screen_width, height=screen_height, bg="#0b081d", bd=0, highlightthickness=0)
        self.canvas.create_image(0, 0, anchor=NW, image=self.photoimg_bg)
        self.canvas.pack(side=TOP)

        self.label = customtkinter.CTkLabel(self.root, text="HumId: Human Identification", font=("Poppins Bold", screen_height // 20), bg_color="#0b081d")
        label_x = screen_width / 2
        label_y = screen_height / 20
        self.label.place(x=label_x, y=label_y, anchor="n")

        # Load images
        img_recognizer = Image.open(r"assets\facerecognizer.jpg")
        img_recognizer = img_recognizer.resize((180, 180), Image.LANCZOS)
        self.photoimg_recognizer = ImageTk.PhotoImage(img_recognizer)

        img_results = Image.open(r"assets\result.jpg")
        img_results = img_results.resize((180, 180), Image.LANCZOS)
        self.photoimg_results = ImageTk.PhotoImage(img_results)

        img_developer = Image.open(r"assets\developer.jpg")
        img_developer = img_developer.resize((180, 180), Image.LANCZOS)
        self.photoimg_developer = ImageTk.PhotoImage(img_developer)

        img_exit = Image.open(r"assets\exit.jpg")
        img_exit = img_exit.resize((180, 180), Image.LANCZOS)
        self.photoimg_exit = ImageTk.PhotoImage(img_exit)

        button_font = ("Poppins Medium", screen_height // 50)

        # Define button positions
        recognizer_x_position = (screen_width // 5) + 25
        results_x_position = (2 * screen_width // 5) - 53
        developer_x_position = (3 * screen_width // 5) - 133 # Slightly adjust to maintain alignment
        exit_x_position = (4 * screen_width // 5) - 210 # Slightly adjust to maintain alignment

        vertical_image_offset = screen_height // 4
        vertical_button_offset = screen_height // 2

        # Place Recognizer elements
        self.label_recognizer = Label(self.root, image=self.photoimg_recognizer, cursor="hand2")
        self.label_recognizer.place(x=recognizer_x_position, y=vertical_image_offset)
        self.label_recognizer.bind("<Button-1>", lambda event: on_image_recognizer_click())

        self.button_recognizer = Button(self.root, text="Image Recognizer", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_image_recognizer_click, width=15, height=1, font=button_font)
        self.button_recognizer.place(x=recognizer_x_position, y=vertical_button_offset)

        # Place Results elements
        self.label_results = Label(self.root, image=self.photoimg_results, cursor="hand2")
        self.label_results.place(x=results_x_position, y=vertical_image_offset)
        self.label_results.bind("<Button-1>", lambda event: on_results_click())

        self.button_results = Button(self.root, text="Saved Images", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_results_click, width=15, height=1, font=button_font)
        self.button_results.place(x=results_x_position, y=vertical_button_offset)

        # Place Developer elements
        self.label_developer = Label(self.root, image=self.photoimg_developer, cursor="hand2")
        self.label_developer.place(x=developer_x_position, y=vertical_image_offset)
        self.label_developer.bind("<Button-1>", lambda event: on_developer_click())

        self.button_developer = Button(self.root, text="Developer", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_developer_click, width=15, height=1, font=button_font)
        self.button_developer.place(x=developer_x_position, y=vertical_button_offset)

        # Place Exit elements
        self.label_exit = Label(self.root, image=self.photoimg_exit, cursor="hand2")
        self.label_exit.place(x=exit_x_position, y=vertical_image_offset)
        self.label_exit.bind("<Button-1>", lambda event: on_exit_click())

        self.button_exit = Button(self.root, text="Exit", cursor="hand2", bg="#1f2833", fg="white", bd=0, command=on_exit_click, width=15, height=1, font=button_font)
        self.button_exit.place(x=exit_x_position, y=vertical_button_offset)

if __name__ == "__main__":
    root = customtkinter.CTk()
    obj = Face_Recognition(root)
    root.mainloop()
