from tkinter import Tk, Toplevel, Label, Button, LabelFrame
from PIL import Image, ImageTk

help_content_col1 = """
User Interface Guide
Buttons:
    - Face Recognizer: Used for detecting and recognizing human faces.
    - Saved Images: View previously saved images.
    - Help: Open this help window.
    - Exit: Close the application.
"""

help_content_col2 = """
    - Ctrl + Q: To Quit Face Recognizing.
FAQs
1. Why is the face recognition not working?
    - Ensure the camera is functioning and the face is well-lit.
2. How do I improve recognition accuracy?
    - Use a high-quality camera and ensure consistent lighting.
"""

def back_to_main(root, dev_window):
    # Destroy the help window
    dev_window.destroy()
    
    # Focus on the main application window
    root.deiconify()  # Make the main application window visible again
    root.focus_set()  # Set focus on the main application window

def help_functionality(root, bg_color="#1f2833"):
    # Hide the main application window while the help window is open
    root.withdraw()

    # Set window size to a larger size to accommodate wider columns
    window_width = 1276
    window_height = 716
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    # Create a new window for help functionality
    dev_window = Toplevel(root)
    dev_window.title("Help")
    dev_window.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
    dev_window.configure(bg=bg_color)  # Set background color

    try:
        # Load help images
        img1 = Image.open("assets/ryan.jpg")  # Replace with the actual path to your image
        img1 = img1.resize((100, 100), Image.LANCZOS)
        photoimg1 = ImageTk.PhotoImage(img1)
        # Load help images
        img2 = Image.open("assets\edmar.jpg")  # Replace with the actual path to your image
        img2 = img2.resize((100, 100), Image.LANCZOS)
        photoimg2 = ImageTk.PhotoImage(img2)
        # Load help images
        img3 = Image.open("assets\carlos.jpg")  # Replace with the actual path to your image
        img3 = img3.resize((100, 100), Image.LANCZOS)
        photoimg3 = ImageTk.PhotoImage(img3)

        # Add widgets and functionality for the help window here
        label = Label(dev_window, text="Help", font=("Poppins Bold", 24), bg=bg_color, fg="white")
        label.grid(row=0, column=0, columnspan=2, pady=10)

        # LabelFrames for help content columns with adjusted width
        col1_frame = LabelFrame(dev_window, text="User Interface Guide", font=("Poppins Bold", 15), bg=bg_color, fg="white", padx=0, pady=5, width=350, height=300)
        col1_frame.grid(row=1, column=0, padx=20, pady=0, sticky="nsew")
        col1_frame.grid_propagate(False)  # Prevent the frame from resizing to fit its content

        # Adjust the position of col2_frame relative to col1_frame
        col2_frame = LabelFrame(dev_window, text="Shortcuts and Tips / FAQs", font=("Poppins Bold", 15), bg=bg_color, fg="white", padx=0, pady=5, width=350, height=300)
        col2_frame.grid(row=1, column=1, padx=20, pady=0, sticky="nsew")
        col2_frame.grid_propagate(False)  # Prevent the frame from resizing to fit its content


        # Help content in LabelFrames
        col1_label = Label(col1_frame, text=help_content_col1, font=("Poppins Regular", 10), bg=bg_color, fg="white", justify="left")
        col1_label.pack(fill="both", expand=True)

        col2_label = Label(col2_frame, text=help_content_col2, font=("Poppins Regular", 10), bg=bg_color, fg="white", justify="left")
        col2_label.pack(fill="both", expand=True)

        # Contact information with images
        contacts = [
            ("Ryan C. Clavo", "ryancana.clavo@bicol-u.edu.ph", photoimg1),
            ("Edmar L. Guevarra", "edmarlavarias.guevarra@bicol-u.edu.ph", photoimg2),
            ("Carlos Miguel O. Rada", "carlosmiguelopena.rada@bicol-u.edu.ph", photoimg3)
        ]

        for i, (name, email, image) in enumerate(contacts):
            image_label = Label(dev_window, image=image, bg=bg_color)
            image_label.grid(row=2+i, column=0, padx=50, pady=5, sticky="w")  # Align to the left (west)
            
            # Display name just to the right of the image
            text_label = Label(dev_window, text=name, font=("Poppins Medium", 12), bg=bg_color, fg="white")
            text_label.grid(row=2+i, column=0, padx=(160, 0), sticky="w")  # Align to the left (west) with some right padding
            
            # Display email just below the name
            email_label = Label(dev_window, text=email, font=("Poppins Regular", 10), bg=bg_color, fg="white")
            email_label.grid(row=2+i, column=0, padx=(160, 0), pady=(70, 5), sticky="w")  # Align to the left (west) with some left padding and some top padding


        # Back button to return to the main application
        back_button = Button(dev_window, text="Back", bg="#385B88", bd=0, font=("Poppins Medium", 12), fg="white", command=lambda: back_to_main(root, dev_window), width=15)
        back_button.grid(row=5, column=0, columnspan=3, pady=20)
        back_button.bind("<Enter>", lambda event: back_button.config(bg="#2C4762"))  # Change color on enter
        back_button.bind("<Leave>", lambda event: back_button.config(bg="#385B88"))  # Change color on leave

        # Centering all columns
        dev_window.grid_columnconfigure(0, weight=1)
        dev_window.grid_columnconfigure(1, weight=1)
        dev_window.grid_rowconfigure(1, weight=1)

    except Exception as e:
        print("Error loading images:", e)

    dev_window.mainloop()

# Example usage
if __name__ == "__main__":
    root = Tk()
    help_functionality(root)
