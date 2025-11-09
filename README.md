**Problem:**  
----

ATM was made to transact for the particular bank accounts but later on the ATMs are connected to interbank network.
The main disadvantage of ATMs are that if the pin is known to anyone, They can use it to take money, so here we implement multifactor user authentication for security purposes.
To reduce the risk of fraudulent activity, several controls can be integrated into the ATM processing environment.
The main use of the Biometric is that it is unique for a person, so we can identify the unauthorized person before they took our money from account.

**Solution Created**:
-----

# Title: Multi-Factor Secure ATM Access with Face Recognition using Deep Learning

A deep learning-based ATM security system that combines Face Recognition, PIN verification, and OTP authentication to provide multi-layer protection against unauthorized access.

>**Overview:**

This project aims to make ATM transactions more secure by using biometric face recognition along with traditional verification methods. The system authenticates users through three layers:

1. Face Recognition – Detects and verifies the user’s face using a trained model.  
1. PIN Verification – Confirms the entered personal identification number.  
1. OTP Verification – Adds an extra security step using one-time passwords.  


>**Technologies Used:**

- Python  
- OpenCV *(Haar Cascade and FisherFace Recognizer)*  
- Streamlit *(for interactive interface)*  
- SMTP *(for sending OTP emails)*  
- Tkinter *(for OTP generation)*  
- NumPy, Pickle, Base64

>**Working**:

1. The system captures the user's image using a webcam.
2. The face is matched with pre-trained data using Haar Cascade.
3. If the face is recognized, the user is asked to enter their PIN.
4. An OTP is generated and sent for the final verification.
5. Access is granted only if all three verifications pass successfully.

>**Results:**

- [X] Successfully detects and recognizes registered faces in real time.  
- [X] Prevents unauthorized ATM access even if PIN or OTP is leaked.  
- [X] Demonstrates how AI and security systems can work together to improve banking safety.

>**Future Enhancements**

1. Replace Haar Cascade with CNN-based model (e.g., FaceNet or DeepFace).
1. Integrate fingerprint or voice recognition.
1. Deploy the model on cloud for real-time banking applications.



