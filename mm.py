import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import random
import time
# facerec.py
import cv2, sys, numpy, os
import urllib
import numpy as np
import time
import os
from subprocess import call
import time
import os
import glob
import smtplib
import base64
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sys
import random
pins=1234
otp_ = random.randint(10000, 100000)

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

        # Construct the email message
        subject = 'OTP and Image'
        # otp_ = random.randint(10000, 100000)
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
            print(f"Error attaching image: {e}")

        # Send the email
        connection.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")
    finally:
        # Close the connection
        connection.close()

# Test the send_email function
# send_email()

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
n=input("enter your name : ")
print('Training...')
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
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
##with open("1.txt", mode='a') as file:
webcam = cv2.VideoCapture(0)

##url="http://192.168.43.1:8080/shot.jpg"
while True:

    (_, im) = webcam.read()
##    imgPath=urllib.urlopen(url)
##    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
##    im=cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:
            #port.write('B')
           # print (names[prediction[0]])
            cv2.putText(im,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            print('The accessing person is ',str(n))
            
            if names[prediction[0]]==n:
                
                print("The detected face person is : ",names[prediction[0]])  
                print('you can proceed your transaction')
                pin=int(input('enter your pin: '))
                if pin ==pins:
                    print("You can continue further")
                    exit()
                
        
                else:
                    send_email()
                    check_otp=int(input("Enter the otp :"))
                    if check_otp==otp_:
                        print("You can continue further")
                        exit()
                    else :
                        print("Incorrect pin ... exiting the portal")
                
                
            else:
                im2=im
                cv2.putText(im2,'unknown ',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
                print("The detected person is unknown ")
                cv2.imwrite('1.jpg',im2)
                send_email()
                check_otp=int(input("enter the otp : "))
                if check_otp==otp_:
                    pin=int(input('enter your pin:'))
                    if pin == pins:
                        print("You can continue further")
                        exit()
                    else:
                        
                        print("Incorrect pin ... exiting the portal")
                        exit()

      
        else:
            cv2.putText(im,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    



















