from rapid_videocr import RapidVideOCR, RapidVideOCRInput
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, LangDet
import glob
import os
from PIL import Image
from copy import deepcopy

linux = True
model_v5_server_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_rec_infer" if linux else "./models/PP-OCRv5_server_rec_infer"
model_v5_server_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_det_infer" if linux else  "./models/PP-OCRv5_server_det_infer"

model_v5_mobile_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_mobile_rec_infer" if linux else "./models/PP-OCRv5_mobile_rec_infer"
model_v5_mobile_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_mobile_det_infer" if linux else "./models/PP-OCRv5_mobile_det_infer"

txt_path = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_rec_infer/ppocrv5_dict.txt" if linux else "./models/PP-OCRv5_server_rec_infer/ppocrv5_dict.txt"
# Document: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4

ocr_input_params = RapidVideOCRInput(
    is_batch_rec=True,
    batch_size=6,
    out_format="srt",
    # Document params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/?h=rec+lang+type
    ocr_params={
        "Global.use_det": True,
        "Global.use_rec": True,
        "Global.use_cls": True,
        "Global.max_side_len": 4000,
        "Rec.model_dir": model_v5_server_rec,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        "Rec.engine_type": EngineType.PADDLE,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.rec_img_shape": [3, 64, 320],

        "Det.model_dir": model_v5_server_det,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        "Det.engine_type": EngineType.PADDLE,
        "Det.lang_type": LangDet.MULTI,
        "Det.model_type": ModelType.SERVER,
        "Det.ocr_version": OCRVersion.PPOCRV5,

        "Det.limit_side_len": 736,
        "Det.limit_type": "min",
        "Det.box_thresh": 0.5,

        "EngineConfig.paddle.use_cuda": True,  # 使用PaddlePaddle GPU版推理
        "EngineConfig.paddle.gpu_id": 0,  # 指定GPU id
        "EngineConfig.paddle.gpu_mem": 8192 if linux else 1024,  # 指定GPU memory
        "Rec.rec_keys_path": txt_path
    }
)
# Version GPU for google colab
# extractor = RapidVideOCR(ocr_input_params)

rgb_dir = "images"
save_dir = "/content/drive/MyDrive/RapidVideoOCR/outputs" if linux else "./outputs"

# save_name = "sub_ocr_new"
target_scale = 2
batch_size = 6
padding = 10

for folder in glob.glob(os.path.join(rgb_dir, "*/")):
    if os.path.isdir(folder):
        folder_name = os.path.basename(os.path.normpath(folder))
        print(f"\nProcessing directory: {folder}\n")
        # outputs/a.srt  outputs/a.txt
        # extractor(folder, save_dir, save_name=folder_name)

        image_files = os.path.join(folder, os.listdir(folder)[0])
        print(image_files)
        if not image_files:
            print(f"No images found in {folder}")
            continue

        with Image.open(image_files) as img:
            w, h = img.size  # h = crop_height (giả sử crop sub dọc)

        limit_side_len = int(target_scale * (batch_size * (h + padding)))
        # Require multiples of 32
        if limit_side_len % 32 != 0:
            limit_side_len = ((limit_side_len // 32) + 1) * 32

        # print(limit_side_len)

        folder_ocr_params = deepcopy(ocr_input_params.ocr_params)
        folder_ocr_params["Det.limit_side_len"] = 736 if limit_side_len < 736 else limit_side_len
        # folder_ocr_params["Global.max_side_len"] = w * target_scale
        # print(folder_ocr_params)

        folder_extractor = RapidVideOCR(
            RapidVideOCRInput(
                is_batch_rec=True,
                batch_size=batch_size,
                out_format="srt",
                ocr_params=folder_ocr_params
            )
        )
        folder_extractor(folder, save_dir, save_name=folder_name)



# folder = "images/name folder"
#
# # Kiểm tra thử folder có tồn tại không
# print("Folder exists:", os.path.exists(folder))
# print("Files in folder:", os.listdir(folder))
