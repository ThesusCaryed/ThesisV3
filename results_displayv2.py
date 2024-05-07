import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk
import datetime
import customtkinter

class ResultsWindow:
    def __init__(self, master):
        self.master = master
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}+0+0")
        self.master.title("Recognized Faces")
        self.master.configure(background='#1f2833')

        # Initialize style
        self.style = ttk.Style()
        self.style.configure('TFrame', background ='#1f2833', borderwidth=2, relief='solid', bordercolor='white')
        self.style.configure('Known.TFrame', background='#293241')
        self.style.configure('Unknown.TFrame', background='#293241')  

        # Customtkinter label
        label_font_size = 40
        self.label = customtkinter.CTkLabel(self.master, text="Recognized Faces", font=("Poppins Bold", label_font_size))
        self.label.pack(pady=30, padx=100, anchor="nw")

        # Sort and Back buttons
        self.sort_descending = False  # Starts as False (ascending)
        self.sort_button_text = tk.StringVar()
        self.sort_button_text.set("Sort Ascending")
        self.sort_button = tk.Button(self.master, textvariable=self.sort_button_text, command=self.toggle_sort, bg="#385B88", fg="white", font=("Poppins", 12), width=15, height=1)
        self.sort_button.place(x=100, y=95)
        self.back_button = tk.Button(self.master, text="Back", command=self.master.destroy, bg="#385B88", fg="white", font=("Poppins", 12), width=15, height=1)
        self.back_button.place(x=270, y=95)

        # Canvas and Scrollbar setup
        self.canvas = tk.Canvas(self.master, bg='#1f2833', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style='TFrame')
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.place(x=screen_width - 20, y=150, height=screen_height - 180)
        self.canvas.place(x=100, y=150, width=screen_width - 120, height=screen_height - 180)

        # Frames for known and unknown images
        self.known_frame = ttk.Frame(self.scrollable_frame, style='Known.TFrame')
        self.unknown_frame = ttk.Frame(self.scrollable_frame, style='Unknown.TFrame')
        self.known_frame.grid(row=1, column=0, sticky="nsew", padx=11, pady=5)
        self.unknown_frame.grid(row=1, column=1, sticky="nsew", padx=11, pady=5)

        # Labels for known and unknown
        self.known_label = tk.Label(self.known_frame, text="Known Faces", font=("Poppins Bold", 14), bg='#293241', fg='white')
        self.unknown_label = tk.Label(self.unknown_frame, text="Unknown Faces", font=("Poppins Bold", 14), bg='#293241', fg='white')
        self.known_label.grid(row=0, column=0, sticky="ew")
        self.unknown_label.grid(row=0, column=0, sticky="ew")

        self.load_images()

    def toggle_sort(self):
        self.sort_descending = not self.sort_descending
        if self.sort_descending:
            self.sort_button_text.set("Sort Descending")
        else:
            self.sort_button_text.set("Sort Ascending")
        self.load_images()

    def show_large_image(self, img_path):
        top = tk.Toplevel(self.master)
        top.title("Image Preview")
        img = Image.open(img_path)
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(top, image=photo)
        img_label.image = photo  # Keep a reference!
        img_label.pack()
        top.geometry(f"{img.width()}x{img.height()}+100+100")

    def load_images(self):
        directory = 'recognized_faces'
        files = os.listdir(directory)
        if not files:
            print("No images to display.")
            return

        full_paths = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files]
        full_paths.sort(key=lambda x: x[1], reverse=self.sort_descending)

        thumbnail_size = (100, 100)
        max_cols = 3
        row_count_known = col_count_known = 0
        row_count_unknown = col_count_unknown = 0

        # Clear previous images if any
        for widget in self.known_frame.winfo_children()[1:]:  # Skip the first child, which is the label
            widget.destroy()
        for widget in self.unknown_frame.winfo_children()[1:]:  # Skip the first child, which is the label
            widget.destroy()

        # Populate frames with new images
        for image_path, _ in full_paths:  # Ignoring mtime now
            img = Image.open(image_path).convert('RGBA')
            img.thumbnail(thumbnail_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            face_name = os.path.splitext(os.path.basename(image_path))[0]

            target_frame = self.unknown_frame if "Unknown" in face_name else self.known_frame
            col_count, row_count = (col_count_unknown, row_count_unknown) if "Unknown" in face_name else (col_count_known, row_count_known)

            frame = ttk.Frame(target_frame)
            frame.grid(row=row_count + 1, column=col_count, padx=5, pady=5, sticky='nsew')  # Start grid from row 1 to keep the label on row 0
            label = tk.Label(frame, image=photo, bg='#1f2833' if target_frame == self.known_frame else '#1f2833', fg='white')
            label.image = photo  # Keep a reference
            label.pack()
            info_label = tk.Label(frame, text=f"{face_name}", bg='#1f2833', fg='white', font=("Arial", 10))
            info_label.pack()
            label.bind("<Button-1>", lambda e, path=image_path: self.show_large_image(path))  # Bind left mouse click

            col_count += 1
            if col_count >= max_cols:
                col_count = 0
                row_count += 1

            if target_frame == self.unknown_frame:
                col_count_unknown, row_count_unknown = col_count, row_count
            else:
                col_count_known, row_count_known = col_count, row_count

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))