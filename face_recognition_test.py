import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import face_recognition
import os
import csv
from datetime import datetime

# Function to load the known images from the "known_people" folder
def load_known_images(known_people_folder):
    known_images = []
    known_encodings = []
    image_names = []  # Store the names of the images
    for filename in os.listdir(known_people_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(known_people_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_images.append(image)
                known_encodings.append(encoding[0])  # Assume one face per image
                image_names.append(filename)  # Store the image name
    return known_encodings, known_images, image_names

# Function to open a file dialog and select the image for comparison
def upload_image():
    clear_images()  # Clear old images when uploading a new one
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if file_path:
        # Show the selected image on the GUI
        uploaded_image = Image.open(file_path)
        uploaded_image = uploaded_image.resize((250, 250))  # Resize for display
        uploaded_image_tk = ImageTk.PhotoImage(uploaded_image)
        image_label.config(image=uploaded_image_tk)
        image_label.image = uploaded_image_tk

        # Compare the uploaded image with known faces
        compare_faces(file_path)

# Function to compare the uploaded image with known images
def compare_faces(uploaded_image_path):
    # Load the known faces from the folder
    known_people_folder = "./known_people"
    known_encodings, known_images, image_names = load_known_images(known_people_folder)

    # Load the uploaded image and encode it
    uploaded_image = face_recognition.load_image_file(uploaded_image_path)
    uploaded_encodings = face_recognition.face_encodings(uploaded_image)

    if not uploaded_encodings:
        messagebox.showerror("Error", "No faces found in the uploaded image.")
        return

    # Loop through the known encodings and compare with the uploaded image
    match_found = False
    matched_image_name = ""
    for i, known_encoding in enumerate(known_encodings):
        results = face_recognition.compare_faces([known_encoding], uploaded_encodings[0], tolerance=0.35)
        if results[0]:
            match_found = True
            matched_image_name = image_names[i]
            break

    # Show the result in the GUI
    if match_found:
        messagebox.showinfo("Result", f"The faces match with: {matched_image_name}")
        show_matched_image(matched_image_name)
        log_match(matched_image_name)  # Log the match and timestamp
    else:
        messagebox.showinfo("Result", "The faces do not match.")

# Function to display the matched image in the GUI
def show_matched_image(matched_image_name):
    known_people_folder = "./known_people"
    image_path = os.path.join(known_people_folder, matched_image_name)
    matched_image = Image.open(image_path)
    matched_image = matched_image.resize((250, 250))  # Resize for display
    matched_image_tk = ImageTk.PhotoImage(matched_image)
    matched_image_label.config(image=matched_image_tk)
    matched_image_label.image = matched_image_tk

# Function to log the matched photo name and timestamp into a CSV file
def log_match(matched_image_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp
    image_name_without_extension = os.path.splitext(matched_image_name)[0]
    log_data = [image_name_without_extension, timestamp]
    
    log_file = 'Attendance_log.csv'  # Define the log file name
    file_exists = os.path.isfile(log_file)
    
    # Open the log file in append mode, create it if it doesn't exist
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header only if the file is empty
        if not file_exists:
            writer.writerow(['photo_matched_name', 'timestamp'])
        
        writer.writerow(log_data)  # Write the match and timestamp to the log

# Function to clear the displayed images
def clear_images():
    image_label.config(image="")
    image_label.image = None
    matched_image_label.config(image="")
    matched_image_label.image = None

# Create the main window
root = tk.Tk()
root.title("Face Recognition GUI")
root.geometry("900x900")

# Add a label to show the uploaded image
image_label = tk.Label(root)
image_label.pack(pady=20)

# Add a label to show the matched image
matched_image_label = tk.Label(root)
matched_image_label.pack(pady=20)

# Add a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
