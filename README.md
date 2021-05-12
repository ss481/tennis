# tennis

1. Arhitecture of tennis prototype
	- Table detecion
	- Text detection
	- Text recognition
	- Serving indicator

2. Table detection model
https://github.com/zhreshold/mxnet-ssd
    The repository is used for training a table dection model.
    10k samples used for training the model.
 (https://drive.google.com/drive/folders/1qip3JOS_enoCGQ6PDFGwNz_VGNwTnHEL?usp=sharing)

3. Text detection model
https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
    Used text detection model from the repository 
 (https://drive.google.com/drive/folders/1lR3fJQIkop08kJp0Abg7MBCoh7yXXPFD?usp=sharing)
 
IMPORTANT!
For the Tesseract-OCR algorithm, path of the installed location is needed.
  input variable: --text-recognition-exe 
   (ex. C:/Program Files/Tesseract-OCR/tesseract.exe)
