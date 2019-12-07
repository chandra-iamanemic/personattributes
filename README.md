# personattributes
## learning pytorch with market 1501 person attribute dataset

1.	Acquired the Market 1501 dataset and the Market Attributes
2.	Reorganized the images from dataset into respective folders based on their ID (data_extraction.py)
3.	 Learnt the basics of PyTorch 
4.	Created a basic CNN network and trained it to learn the Gender Attribute of the Market dataset Images. (On CPU) (gender_classifier.py)
5.	Trained a resnet34 model to learn the Gender Attribute of the Market dataset Images. (On GPU without freezing any layers) (gender_classifier_gpu_resnet34_no_freeze.py)
6.	Trained a resnet34 model to learn the Gender Attribute of the Market dataset Images. (On GPU by freezing initial layers) (gender_classifier_gpu_resnet34_with_freeze.py)
7.	Trained a resnet34 model to learn the Gender and Backpack Attributes of the Market dataset Images. (gender_backpack_trainer.py)
8.	Trained a resnet34 model to learn the Gender, Backpack and Age Attributes of the Market dataset Images. (gender_backpack_age_trainer.py)
