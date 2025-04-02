# Gender and Age Prediction using Keras & TensorFlow  

## Overview  
A CNN-based model for **gender and age prediction** from facial images using **Keras** and **TensorFlow**.  

## Features  
- Predicts **gender** (Male/Female) and **age** from images  
- Uses **CNNs** for feature extraction  
- Supports **training, testing, and real-time prediction**  

## Requirements  
Install dependencies:  
```bash
pip install tensorflow keras numpy opencv-python matplotlib
```

## Usage  
- **Train the model**:  
  ```bash
  python Gender_and_Age_Prediction.py --train
  ```
- **Test on an image**:  
  ```bash
  python Gender_and_Age_Prediction.py --image path/to/image.jpg
  ```
- **Real-time prediction**:  
  ```bash
  python Gender_and_Age_Prediction.py --webcam
  ```

## Future Scope  
- Improve accuracy with **transfer learning**  
- Enhance dataset diversity  

---

