from rapid_videocr import RapidVideOCR, RapidVideOCRInput
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, LangDet
import glob
import os

model_v5_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_rec_infer"
model_v5_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_det_infer"
txt_path = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_rec_infer/ppocrv5_dict.txt"
# Document: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4

ocr_input_params = RapidVideOCRInput(
    is_batch_rec=True,
    batch_size=6,
    out_format="srt",
    # Document params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/?h=rec+lang+type
    ocr_params={
        "Rec.model_dir": model_v5_rec,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        "Det.model_dir": model_v5_det,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        "Rec.engine_type": EngineType.PADDLE,
        "Det.engine_type": EngineType.PADDLE,
        "Rec.lang_type": LangRec.JAPAN,
        "Det.lang_type": LangDet.MULTI,
        "Rec.model_type": ModelType.SERVER,
        "Det.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "EngineConfig.paddle.use_cuda": True,  # 使用PaddlePaddle GPU版推理
        "EngineConfig.paddle.gpu_id": 0,  # 指定GPU id
        "EngineConfig.paddle.gpu_mem": 12288,  # 指定GPU memory
        "Rec.rec_keys_path": txt_path
    }
)
# Version GPU for google colab
extractor = RapidVideOCR(ocr_input_params)

rgb_dir = "images"
save_dir = "/content/drive/MyDrive/RapidVideoOCR/outputs"

# save_name = "sub_ocr_new"

for folder in glob.glob(os.path.join(rgb_dir, "*/")):
    if os.path.isdir(folder):
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"\nProcessing directory: {folder}\n")
        # outputs/a.srt  outputs/a.txt
        extractor(folder, save_dir, save_name=folder_name)
