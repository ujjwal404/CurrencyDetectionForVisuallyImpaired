# Currency Detection for Visually Impaired 👨‍🦯
This repo contains code for the CNN model that can be deployed on a RaspberryPi and used for currency detection. [Here](https://youtube.com/shorts/U-Lis5e01yQ?feature=share)'s a demo video of final result. 

IEEE citation can be found [here](https://ieeexplore.ieee.org/document/10458734).

# Directory details:
- `build_dataset.py` : Crops all the images into a defined size and splits the dataset into test,train and validation.
- `train.py`: Extracts list of labels from image names and makes use of input_fn.py and model_fn.py.
- `model/input_fn.py`: Creates a dataframe with images and labels and defined batches. All the parameters can be set in `model/params.json`.
- `model/model_fn.py`: Uses make_model to pass data through the CNN and saves the model at the end.

- Remaining files are utitlities that were used for experimentation and model conversion.

# Model Architecture:
![Screenshot 2022-12-29 at 8 16 28 PM](https://user-images.githubusercontent.com/63122405/209969512-432bc780-c919-4779-855b-7830d690c984.png)

# Result:
1. Train Accuracy: 96.60314830157415
2. Validation Accuracy : 92.12924606462303 
3. Test Accuracy : 92.79801324503312

# Running on hardware
After setting up RaspberryPi OS, the model can be deployed and run using the script [here](https://drive.google.com/drive/folders/1XLfEu91L61ctRc5iKFSAE-7q4bhZwy5j?usp=sharing). In RaspberryPi terminal run:
```
Python3 script.py --labels labels.txt --model model.tflite
```

# Dataset 
The dataset contains 7 different classes of INR notes and can be downloaded from [here](https://data.mendeley.com/datasets/8ckhkssyn3).



This project was created by [me](https://ujjwalkadam.tech/about), [Arvind](https://github.com/AvDjah), [Chaudhary Abuzar](chaudhary.cs19@nsut.ac.in) and [Ujjwal](https://github.com/ujjwal2604) as B.Tech. Project, year 2022-23. 
