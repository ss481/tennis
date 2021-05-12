1. Arhitecture of tennis prototype
	- Table detecion
	- Text detection
	- Text recognition
	- Serving indicator

2. Table detection model
https://github.com/zhreshold/mxnet-ssd
    The repository is used for training a table dection model.
    10k samples used for training the model.
	
3. Text detection model
https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
    Used text detection model from the repository 


IMPORTANT!
For the Tesseract-OCR algorithm, path of the installed location is needed.
  input variable: --text-recognition-exe 
   (ex. C:/Program Files/Tesseract-OCR/tesseract.exe)