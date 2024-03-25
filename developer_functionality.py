from tkinter import Tk, Toplevel, Label, Button
from PIL import Image, ImageTk

def back_to_main(root, dev_window):
    # Destroy the developer window
    dev_window.destroy()
    
    # Focus on the main application window
    root.deiconify()  # Make the main application window visible again
    root.focus_set()  # Set focus on the main application window

def developer_functionality(root, bg_color="#1f2833"):
    # Hide the main application window while the developer window is open
    root.withdraw()

    # Create a new window for developer functionality
    dev_window = Toplevel(root)
    dev_window.title("Developer")
    dev_window.geometry("700x400")
    dev_window.configure(bg=bg_color)  # Set background color

    try:
        # Load developer images
        img1 = Image.open("assets/2.png")  # Replace "path_to_image1.jpg" with the actual path to your first image
        img1 = img1.resize((200, 200), Image.LANCZOS)
        photoimg1 = ImageTk.PhotoImage(img1)

        # Add widgets and functionality for the developer window here
        label = Label(dev_window, text="Developer", font=("Poppins Bold", 20), bg=bg_color, fg="white")
        label.grid(row=0, column=1, columnspan=3, pady=10)

        # Image 1 and text
        image_label1 = Label(dev_window, image=photoimg1, bg=bg_color)
        image_label1.grid(row=1, column=1, padx=(15, 10), pady=(50, 0), sticky="nsew")
        text_label1 = Label(dev_window, text="Ryan C. Clavo", font=("Poppins Medium", 14), bg=bg_color, fg="white")
        text_label1.grid(row=2, column=1, pady=1)

        # Add additional text label under text_label1
        additional_text1 = Label(dev_window, text="ryancana.clavo@bicol-u.edu.ph", font=("Poppins Regular", 10), bg=bg_color, fg="white")
        additional_text1.grid(row=3, column=1, pady=2)

        # Image 2 and text
        image_label2 = Label(dev_window, image=photoimg1, bg=bg_color)
        image_label2.grid(row=1, column=2, padx=(20, 10), pady=(50, 0), sticky="nsew")
        text_label2 = Label(dev_window, text="Edmar L. Guevarra", font=("Poppins Medium", 14), bg=bg_color, fg="white")
        text_label2.grid(row=2, column=2, pady=10, padx=5)

        # Add additional text label under text_label2
        additional_text2 = Label(dev_window, text="edmarlavarias.guevarra@bicol-u.edu.ph", font=("Poppins Regular", 10), bg=bg_color, fg="white")
        additional_text2.grid(row=3, column=2, pady=10)

        # Image 3 and text
        image_label3 = Label(dev_window, image=photoimg1, bg=bg_color)
        image_label3.grid(row=1, column=3, padx=(20, 10), pady=(50, 0), sticky="nsew")
        text_label3 = Label(dev_window, text="Carlos Miguel O. Rada", font=("Poppins Medium", 14), bg=bg_color, fg="white", )
        text_label3.grid(row=2, column=3, pady=5)

        # Add additional text label under text_label3
        additional_text3 = Label(dev_window, text="carlosmiguelopena.rada@bicol-u.edu.ph", font=("Poppins Regular", 10), bg=bg_color, fg="white")
        additional_text3.grid(row=3, column=3, pady=5)

        # Back button to return to the main application
        back_button = Button(dev_window, text="Back", bg="#385B88", bd=0, font=("Poppins Medium", 12), fg="white", command=lambda: back_to_main(root, dev_window), width=15)  # Increase width here
        back_button.grid(row=5, column=1, columnspan=3, pady=10)
        back_button.bind("<Enter>", lambda event: back_button.config(bg="#2C4762"))  # Change color on enter
        back_button.bind("<Leave>", lambda event: back_button.config(bg="#385B88"))  # Change color on leave

        # Centering all columns
        dev_window.grid_columnconfigure((0, 4), weight=1)

    except Exception as e:
        print("Error loading images:", e)

    dev_window.mainloop()

# Example usage
if __name__ == "__main__":
    root = Tk()
    developer_functionality(root)