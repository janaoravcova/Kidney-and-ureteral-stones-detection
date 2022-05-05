# Kidney-and-ureteral-stones-detection
This code is the implementation of two proposed solutions for my diploma thesis. 
The topic of the thesis is detection of kidney stones in CT images and link to text will be provided later.

## 3DCC
### Data
Data was provided from private source, therefore can't be shared in any way. Model was trained on non-contrast abdominal CT scans

### Detection
To run detection, run detection.py. Specify path to pretrained model and data in code.

### Extraction of candidates
The first step - extracting candidates from CT volume is performed beforehand to make the training and tuning easier. 
To extract candidates run use method main(path) in utils.py with path to your dataset. Extracted candidates will be saved in train/ (val/) folders. 
Labels will be in root directory.

### Training
To train the binary classification of extracted candidates, use main.py file, where you can specify the paths to labels generated in previous step. 

### Model
Details of the model are available in file model.py

### Dataloader
Loading the extracted candidates is described by dataset.py
