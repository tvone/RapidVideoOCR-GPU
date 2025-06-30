from rapid_videocr import RapidVideOCR, RapidVideOCRInput
from rapidocr import EngineType, LangRec, ModelType, OCRVersion
import glob
import os

model_v3 = "models/japan_PP-OCRv3_rec_infer.onnx"
model_v4 = "models/japan_PP-OCRv4_rec_infer.onnx"
model_v5 = "models/PP-OCRv5_server_rec_infer.onnx"

# Document: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4
ocr_input_params = RapidVideOCRInput(
   is_batch_rec=False,
   ocr_params={
        "Rec.model_path": model_v5,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
   }
)
extractor = RapidVideOCR(ocr_input_params)

rgb_dir = "images"
save_dir = "outputs"
# save_name = "sub_ocr_new"

for folder in glob.glob(os.path.join(rgb_dir, "*/")):
    if os.path.isdir(folder):
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"\nProcessing directory: {folder}\n")
        # outputs/a.srt  outputs/a.txt
        extractor(folder, save_dir, save_name=folder_name)



