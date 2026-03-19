## Fruit/Vegetable Prediction Script

Using the fine-tuned CatBoost model, the goal is to classify unseen image data as fruit or vegetable. 

## Usage

```
git clone https://github.com/dd080604/Live-Project-HW-3.git
```

```
%cd Live-Project-HW-3
pip install -r requirements.txt
```

```
python scripts/inference.py path/to/images --output predictions.csv
```
Input the path to the folder of images where it says 'path/to/images'. Running these commands will output a CSV file containing the file object name and its predicted class. 
