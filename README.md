"pip install uv -q" and "uv sync" to download all the necessary library

replace the modeling_donut_swin.py of your downloaded library by the modeling_donut_swin.py file at current working directory. For example:
cp modeling_donut_swin.py D:\.vscode\testPMDRL\.venv\Lib\site-packages\transformers\models\donut\modeling_donut_swin.py
Dataset:
https://drive.google.com/file/d/1DeFltqICrB9WqTTz_ZX1QtAfxqjIGG_3/view?usp=sharing
download anh put the folder in to working directory
Weight: 
https://drive.google.com/file/d/1z0CBiGQXsuGqSfY95p01H1RozrYASf8d/view?usp=sharing
Download and put folder in to working directory and "uv run app.py" to test 3 model

Train:(the config is match )
uv run TrOCR_train.py
uv run Donut_train.py
uv run YOLO.py

Online Training on Google Collab: 
folowing the Captcha_Reader.ipynb

To create a new dataset ,you just need to modify the metadata of dataset at config.py then run you can run sample.py to watch some images and run generate_dataset.py to generate captcha_dataset