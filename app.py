
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from flask import Flask,render_template,request


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_caption(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds



app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1
@app.route('/')
def index():
  return render_template('index.html')
@app.route('/after',methods=['GET','POST'])
def after():
  file=request.files['file1']
  file.save('static/file.jpg')
  L_cap=predict_caption(['static/file.jpg'])
  caption=''
  for i in L_cap:
    caption+=i
    

  return render_template('predict.html',final=caption)




if __name__=="__main__":
  app.run(debug=True)

