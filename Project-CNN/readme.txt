This project aims to recognize house number digits in a given image.
It uses a two-stage method:
1- MSER to detect ROIs 
2- A CNN model to classify the ROIs in the image.

Files descriptoin:
1- models.py:
This python file contains the models used for classification digits in a given image.
It contains a custom model as well as VGG16 classes 
2- util.py: 
This file contains helper function to preprocess and load Google SVHN dataset. It is being used in train_model.py
3- train_model.py 
This file contains the neccessary codes to:
    - Load the preprocess the dataset and save them in a loader for training 
    - Training code for the CNN models
4- recognize_house_number.py 
This file is used to recognize house number in a given images. It has codes to:
    - MSER implementatoin to detect ROIs
    - Non-maximum Supperssion to get rid of unnecessary boudning boxes 
    - Inferece for the CNN network. 

5-run.py:
This file process images in folder called "test_images" and output the result in a folder named "graded_images". 
The folder must be contains in the same folder as all the python code for this project. 

Usage of the files: 

-To train the models:
    python train_model.py --model_type "VGG16" --pretained "True" --out_weights_file "vgg_pretrained.pt"  --pickle_file "loaders.pickle"

    --model_type: "custom model" or "VGG16"
    --pretained: "True" or "False"
    --out_weights_file: the desired model name to be saved 
    --pickle_file: (if available) The pickle file for that data loaders 

To use run.py: 
    python run.py --model_type "VGG16" --weights_file "vgg_pretrained.pt"

    --model_type: "custom model" or "VGG16"
    --weights_file: path for the trained model's weights 

To test one image, use  recognize_house_number.py:
    python recognize_house_number.py --model_type "VGG16" --weights_file "vgg_pretrained.pt" --input_image image_path --out_image out_image_path






