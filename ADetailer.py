from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import math
from PIL import Image, ImageDraw, ImageFilter
import PIL
import random


#download
## !pip install huggingface_hub
## !pip install ultralytics

class ADetailerDiffuser:
  MODELS = ["face_yolov8n_v2.pt","face_yolov8m.pt","hand_yolov8n.pt"]

  def __init__(self, model, fidelity):
    if model not in self.MODELS:
      assert model+" isn't an accettable model"

    path = hf_hub_download("Bingsu/adetailer", model)
    self.reconizer = YOLO(path)
    self.fidelity = fidelity




  def reconize(self, img):
    output = self.reconizer(img)
    out = []
    for r in output:
        boxes = r.boxes
        for box in boxes:
            if(box.conf>self.fidelity):
              b = box.xyxy[0]
              c = box.cls
              out.append([b, self.reconizer.names[int(c)]])

    return out

  def prepareImage(self, img, out, width, height, percent):
    w, h = img.size
    wo = int(w * percent)
    ho = int(h * percent)
    masks = []

    

    for face in out:
      shape = (int(face[0][0]-wo),int(face[0][1]-ho), int(face[0][2]+wo),int(face[0][3]+ho))
      
      # creating new Image object
      temp_img = img.crop(shape)
      w_t, h_t = temp_img.size
      width_n, height_n = (0,0)


      if(w_t>h_t):
        width_n = width
        height_n = int((h_t/w_t)*width_n)
      else:
        height_n = height
        width_n = int((w_t/h_t)*height_n)
        
      temp_img = temp_img.resize((width_n,height_n), Image.Resampling.LANCZOS)


      masks.append(temp_img)
    return masks

  def __call__(self, img:Image, args):
    if "step" not in args.keys():
        args["step"] = 28
    if "denoise" not in args.keys():
        args["denoise"] = 0.5
    if "positive" not in args.keys():
        args["positive"] = "highly detailed face"
    if "negative" not in args.keys():
        args["negative"] = ""
    if "width" not in args.keys():
        args["width"] = 512
    if "height" not in args.keys():
        args["height"] = 512
    if "cfg" not in args.keys():
        args["cfg"] = 7.5
    if "percent" not in args.keys():
        args["percent"] = 1
    if "blur_radius" not in args.keys():
        args["blur_radius"] = 5
    if "cb" not in args.keys():
        args["cb"] = None
    if "dilatation" not in args.keys():
    	args["dilatation"] = 0
    percent = args["percent"]/1000

    print(percent)

    out = self.reconize(img)
    images = self.prepareImage(img, out, args["width"], args["height"], percent)

    dilatation = args["dilatation"]
    out_img = img.copy()
    w, h = img.size
    wo = int(w * percent)
    ho = int(h * percent)

    def mask_circle_solid(pil_img, blur_radius, wo,ho, dilatation):
      background = Image.new(pil_img.mode, pil_img.size, "#000000")

      mask = Image.new("L", pil_img.size, 0)
      draw = ImageDraw.Draw(mask)
      draw.ellipse([(wo-dilatation,ho-dilatation),(pil_img.size[0]-wo+dilatation, pil_img.size[1]-ho+dilatation)], fill=255)
      mask = mask.filter(PIL.ImageFilter.GaussianBlur(blur_radius))
      #display(mask)
      pil_img.putalpha(mask)
      return pil_img

    for im,[box,_] in zip(images,out):
      #display(im)
      
      if not(args["ip_adap_en"]):
        image_generate = args["pipe"].img2img(image = im, prompt = args["positive"], negative_prompt = args["negative"], num_inference_steps = args["step"], strength = args["denoise"], guidance_scale = args["cfg"], callback=args["cb"]).images[0]
      else:
        image_generate = args["pipe"].generate(image = im, prompt = args["positive"], negative_prompt = args["negative"], num_inference_steps = args["step"], num_samples=1, strength = args["denoise"], guidance_scale = args["cfg"], callback=args["cbf"], callback_steps=1, face_image=args["ip_adap_face_image"], faceid_embeds=args["ip_adap_faceid_embeds"], scale=args["ip_adap_scale"], s_scale=args["ip_adap_scale"],noise=args["ip_adap_scale"], shortcut=True )[0]
      #display(image_generate)
      image_generate = image_generate.resize((int(box[2]-box[0]+2*wo),int(box[3]-box[1]+2*ho)), Image.Resampling.LANCZOS)
      image_generate = mask_circle_solid(image_generate, args["blur_radius"], wo,ho,dilatation)
      
      print(wo,ho)
      out_img.paste(image_generate, (int(box[0]-wo), int(box[1]-ho)), image_generate)

    return out_img