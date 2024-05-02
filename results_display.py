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

        # Buttons setup
        button_height = 30
        button_y_position = 60
        button_padding_below = 10

        # Font sizes optimized for laptop screens
        label_font_size = max(screen_height // 30, 30)
        button_font_size = max(screen_height // 150, 10)

        # Add label at the top with customtkinter
        self.label = customtkinter.CTkLabel(self.master, text="Recognized Faces", font=("Poppins Bold", label_font_size))
        self.label.pack(pady=20, padx=150, anchor="nw")

        # Sort and Back buttons
        self.sort_button = tk.Button(self.master, text="Sort Ascending", command=self.toggle_sort, bg="#385B88", fg="white", font=("Poppins", button_font_size), width=15, height=1)
        self.sort_button.place(x=160, y=80)
        self.sort_button.bind("<Enter>", lambda event: self.on_enter(self.sort_button, "#2C4762"))
        self.sort_button.bind("<Leave>", lambda event: self.on_leave(self.sort_button, "#385B88"))

        self.back_button = tk.Button(self.master, text="Back", command=self.master.destroy, bg="#385B88", fg="white", font=("Poppins", button_font_size), width=15, height=1)
        self.back_button.place(x=300, y=80)
        self.back_button.bind("<Enter>", lambda event: self.on_enter(self.back_button, "#2C4762"))
        self.back_button.bind("<Leave>", lambda event: self.on_leave(self.back_button, "#385B88"))

        # Canvas and Scrollbar setup
        self.canvas = tk.Canvas(self.master, bg='#1f2833', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style='My.TFrame')

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.place(x=screen_width - 20, y=95, height=screen_height - 100)
        self.canvas.place(x=130, y=130, width=screen_width - 120, height=screen_height - 100)

        self.sort_descending = False
        self.load_images()

        # Apply styles for ttk elements
        style = ttk.Style()
        style.configure('My.TFrame', background='#1f2833')

    def on_enter(self, widget, color):
        widget['background'] = color

    def on_leave(self, widget, color):
        widget['background'] = color

    def toggle_sort(self):
        self.sort_descending = not self.sort_descending
        self.load_images()

    def load_images(self):
        directory = 'recognized_faces'
        try:
            files = os.listdir(directory)
            full_paths = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files]
            full_paths.sort(key=lambda x: x[1], reverse=self.sort_descending)

            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            thumbnail_size = (100, 100)  # You might adjust this if necessary
            col_count = 0
            row_count = 0
            max_cols = 6

            for i in range(max_cols):
                self.scrollable_frame.columnconfigure(i, weight=1, uniform="group1")

            for image_path, mtime in full_paths:
                img = Image.open(image_path).convert('RGBA')
                img.thumbnail(thumbnail_size, Image.LANCZOS)  # Use LANCZOS for high-quality downsampling
                img = img.resize(thumbnail_size, Image.LANCZOS)  # Use LANCZOS for high-quality resampling

                photo = ImageTk.PhotoImage(img)
                date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                face_name = os.path.splitext(os.path.basename(image_path))[0]

                frame = ttk.Frame(self.scrollable_frame, style='My.TFrame')
                frame.grid(row=row_count, column=col_count, padx=10, pady=10, sticky='nsew')

                label = tk.Label(frame, image=photo, bg='#1f2833')
                label.image = photo  # Keep a reference
                label.pack(fill='both', expand=True)

                info_text = f"{face_name}\n{date_str}"
                info_label = tk.Label(frame, text=info_text, bg='#1f2833', fg='white', font=("Poppins", 8))
                info_label.pack(fill='x')

                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
                    row_count += 1

                label.bind("<Button-1>", lambda e, path=image_path: self.open_image(path))

            if self.sort_descending:
                self.sort_button.config(text="Sort Ascending")
            else:
                self.sort_button.config(text="Sort Descending")

        except FileNotFoundError:
            print("The directory does not exist")

    def open_image(self, path):
        top = tk.Toplevel(self.master)
        top.title("Image View")
        top.configure(background='#1f2833')
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo, bg='#1f2833')
        label.image = photo
        label.pack()

def main():
    root = tk.Tk()
    app = ResultsWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
