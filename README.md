🔧 Requirements

Python: 3.10.0

NumPy: 2.2.6

MediaPipe: 0.10.11

pip install numpy==2.2.6 mediapipe==0.10.11

📂 Training Data

All trained gesture data is stored in the file:

templates.npy

🧠 Training the Model

To train the gesture data, run the following command:

python main.py --setup


This will capture gestures from the camera and save the training data into templates.npy.

▶️ Running the Project

To start the gesture recognition system, use:

python main.py --run


The application will open the camera and display text output for the recognized gestures.

❌ Exit the Application

Press q to safely close the application.
