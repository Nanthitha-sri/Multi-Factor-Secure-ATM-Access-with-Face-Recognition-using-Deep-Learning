import streamlit as st
import cv2
import numpy as np
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import os
import pickle
import base64


#========================  BACKGROUND IMAGE ===========================


st.markdown(f'<h1 style="color:#ffffff ;text-align: center;font-size:34px;font-family:canvet;">{"Real-Time ATM Face Recognization"}</h1>', unsafe_allow_html=True)
st.write("-------------------------------------------")



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')

# # Global Variables
pins = 1234


# import pickle
# with open('otpp.pickle', 'wb') as f:
#     pickle.dump(otp_, f)


def gen_otp():


    import tkinter as tk
    import random
    import pickle
    
    # Function to generate a random number and save it
    def generate_and_save():
        random_number = random.randint(1, 100)  # Generate a random number between 1 and 100
        if random_number == []:
            with open('random_number.pkl', 'rb') as f:
                random_number = pickle.load(f)  
        else:
            
            with open('random_number.pkl', 'wb') as f:
                pickle.dump(random_number, f)  # Save the number as a pickle file
            # result_label.config(text=f"Generated Number: {random_number}")
        
    # Create the main window
    root = tk.Tk()
    root.title("Random Number Generator")
    
    # Add a button to generate the random number
    generate_button = tk.Button(root, text="Generate Random Number", command=generate_and_save)
    generate_button.pack(pady=20)
    
    # Label to display the result
    # result_label = tk.Label(root, text="Generated Number: None")
    # result_label.pack(pady=10)
    
    # Run the Tkinter event loop
    root.mainloop()




        
with open('random_number.pkl', 'rb') as f:
    otp_ = pickle.load(f)   



haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Function to send OTP email
def send_email():
    # Email configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465  # For SSL
    sender_email = 'xxxxxxxxxxxxxxxxxxxxxxx'  # replace your mail
    receiver_email = 'xxxxxxxxxxxxxxxxxxxxxxx'  # replace your mail
    password = 'App password'  # Your email password

    # Create a secure SSL connection to the SMTP server
    connection = smtplib.SMTP_SSL(smtp_server, smtp_port)

    try:
        # Login to the email server
        connection.login(sender_email, password)
        
        
        with open('random_number.pkl', 'rb') as f:
            otp_ = pickle.load(f)   

        # Construct the email message
        subject = 'OTP and Image'
        body = "Your OTP for logging in: " + str(otp_)

        # Create a MIMEMultipart message to support both text and attachments
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Attach the OTP message as plain text
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image
        try:
            with open("1.jpg", 'rb') as fp:
                img = MIMEImage(fp.read())
                msg.attach(img)
        except Exception as e:
            st.error(f"Error attaching image: {e}")

        # Send the email
        connection.sendmail(sender_email, receiver_email, msg.as_string())
        st.success("Email sent successfully!")

    except Exception as e:
        st.error(f"Error sending email: {e}")
    finally:
        # Close the connection
        connection.close()

# Function to train the face recognizer
def train_model():
    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # Train the face recognizer model
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(images, labels)

    return model, names

# Streamlit Interface
# st.title("ATM Face Recognization")

# User name input
n = st.text_input("Enter your name:", "")

if n:
    # Train the model once
    model, names = train_model()

    # Webcam Input using Streamlit camera input widget
    st.write("### Click 'Capture Video' to start the camera feed")

    # Using Streamlit's camera input widget to capture video
    camera_input = st.camera_input("Capture", key="camera")

    if camera_input is not None:
        # Convert the image from bytes to an OpenCV-compatible format
        img_bytes = camera_input.getvalue()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load face cascade
        face_cascade = cv2.CascadeClassifier(haar_file)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))

            # Try to recognize the face
            prediction = model.predict(face_resize)

            # Draw rectangle and display name on face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 500:
                cv2.putText(img, names[prediction[0]], (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                if names[prediction[0]] == n:
                    st.write(f"The detected person is {names[prediction[0]]}")
                    pin = int(st.text_input('Enter your pin:'))

                    if pin == pins:
                        st.success("You can continue further")
                    else:
                        send_email()
                        check_otp = int(st.text_input("Enter the OTP:"))
                        with open('random_number.pkl', 'rb') as f:
                            otp_ = pickle.load(f)
                        if int(check_otp) == int(otp_):
                            st.success("OTP matched! You can continue.")
                            pin = st.text_input('Enter your pin:')
                            if int(pin) == pins:
                                st.success("You can continue further")
                            else:
                                st.error("Incorrect pin ... exiting the portal")
                        else:
                            st.error("Incorrect OTP ... exiting the portal")
                else:
                    st.write("The detected person is unknown")
                    gen_otp()

                    # cv2.imwrite('1.jpg', img)
                    send_email()
                    check_otp = int(st.text_input("Enter the OTP:"))
                    
                    # st.write(check_otp)
                    # st.write(otp_)
                    
                    
                    if check_otp == otp_:
                        # st.write(check_otp)
                        # st.write(otp_)

                        pin = st.text_input('Enter your pin:')
                        if int(pin) == pins:
                            st.success("You can continue further")
                        else:
                            st.error("Incorrect pin ... exiting the portal")
                    else:
                        st.error("OTP is Invalid")
                        
            else:
                st.write("No face detected. Please try again.")
        
        # Display the captured image with faces detected
        # im2 =( cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        cv2.imwrite('1.jpg',img)

        # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
