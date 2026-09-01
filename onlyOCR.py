import sys
# Support import in google colab
sys.path.append('/content')
from RapidVideOCR.rapid_videocr import RapidVideOCR, RapidVideOCRInput
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, LangDet
import glob
import os
from PIL import Image
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
import zipfile

class OCR:
    def __init__(self, linux: bool, is_batch_rec: bool, batch_size: int = 6, engine_type: str = 'paddle'):
        self.linux = linux
        self.is_batch_rec = is_batch_rec
        self.batch_size = batch_size

        self.model_v6_server_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv6_medium_rec" if self.linux else "./models/PP-OCRv6_medium_rec"
        self.model_v6_server_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv6_medium_det" if self.linux else "./models/PP-OCRv6_medium_det"

        self.model_v6_server_rec_onnx = "/content/RapidVideoOCR-GPU/models/PP-OCRv6_medium_rec_onnx/inference.onnx" if self.linux else "./models/PP-OCRv6_medium_rec_onnx/inference.onnx"
        self.model_v6_server_det_onnx = "/content/RapidVideoOCR-GPU/models/PP-OCRv6_medium_det_onnx/inference.onnx" if self.linux else "./models/PP-OCRv6_medium_det_onnx/inference.onnx"

        self.model_v5_server_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_rec_infer" if self.linux else "./models/PP-OCRv5_server_rec_infer"
        self.model_v5_server_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_server_det_infer" if self.linux else  "./models/PP-OCRv5_server_det_infer"

        self.model_v5_mobile_rec = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_mobile_rec_infer" if self.linux else "./models/PP-OCRv5_mobile_rec_infer"
        self.model_v5_mobile_det = "/content/RapidVideoOCR-GPU/models/PP-OCRv5_mobile_det_infer" if self.linux else "./models/PP-OCRv5_mobile_det_infer"

        self.txt_path_v5 = "/content/RapidVideoOCR-GPU/models/dict/ppocrv5_dict.txt" if self.linux else "./models/dict/ppocrv5_dict.txt"
        self.txt_path_v6 = "/content/RapidVideoOCR-GPU/models/dict/ppocrv6_dict.txt" if self.linux else "./models/dict/ppocrv6_dict.txt"

        # self.ocr_input_params = {
        #     "is_batch_rec": self.is_batch_rec,
        #     batch_size: self.batch_size,
        #     "out_format": "srt",
        #
        #     # Document params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/?h=rec+lang+type
        #     "ocr_params": {
        #         "Global.use_det": True,
        #         "Global.use_rec": True,
        #         "Global.use_cls": False,
        #         "Global.max_side_len": 4000,
        #         "Rec.model_dir": self.model_v5_server_rec,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        #         "Rec.engine_type": EngineType.PADDLE,
        #         "Rec.lang_type": LangRec.JAPAN,
        #         "Rec.model_type": ModelType.SERVER,
        #         "Rec.ocr_version": OCRVersion.PPOCRV5,
        #         "Rec.rec_img_shape": [3, 48, 320],
        #
        #         "Det.model_dir": self.model_v5_server_det,  # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
        #         "Det.engine_type": EngineType.PADDLE,
        #         "Det.lang_type": LangDet.MULTI,
        #         "Det.model_type": ModelType.SERVER,
        #         "Det.ocr_version": OCRVersion.PPOCRV5,
        #
        #         "Det.limit_side_len": 1280,
        #         "Det.limit_type": "max",
        #         "Det.box_thresh": 0.5,
        #
        #         "EngineConfig.paddle.use_cuda": True,  # 使用PaddlePaddle GPU版推理
        #         "EngineConfig.paddle.gpu_id": 0,  # 指定GPU id
        #         "EngineConfig.paddle.gpu_mem": 14336 if self.linux else 1024,  # 指定GPU memory
        #         "Rec.rec_keys_path": self.txt_path_v5
        #     }
        # }

        self.ocr_input_params = {
            "is_batch_rec": self.is_batch_rec,
            "batch_size": self.batch_size,
            "out_format": "srt",

            # Document params: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/parameters/?h=rec+lang+type
            "ocr_params": {
                "Global.use_det": True,
                "Global.use_rec": True,
                "Global.use_cls": False,
                "Global.max_side_len": 4000,

                "Rec.model_dir": self.model_v6_server_rec, # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
                "Rec.engine_type": EngineType.PADDLE,
                "Rec.lang_type": LangRec.JAPAN,
                "Rec.model_type": ModelType.MEDIUM,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.rec_img_shape": [3, 48, 320],

                "Det.model_dir": self.model_v6_server_det, # model_dir for paddlepaddle-gpu, if it diffirent will be model_path
                "Det.engine_type": EngineType.PADDLE,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MEDIUM,
                "Det.ocr_version": OCRVersion.PPOCRV6,

                "Det.limit_side_len": 1280,
                "Det.limit_type": "max",
                "Det.box_thresh": 0.5,

                "EngineConfig.paddle.use_cuda": True,  # 使用PaddlePaddle GPU版推理
                "EngineConfig.paddle.cuda_ep_cfg.device_id": 0,  # 指定GPU id
                "EngineConfig.paddle.cuda_ep_cfg.gpu_mem": 14336 if self.linux else 1024,  # 指定GPU memory

                "Rec.rec_keys_path": self.txt_path_v6
            } if engine_type == "paddle" else
            {
                "Global.use_det": True,
                "Global.use_rec": True,
                "Global.use_cls": False,
                "Global.max_side_len": 4000,

                "Rec.model_path": self.model_v6_server_rec_onnx,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.JAPAN,
                "Rec.model_type": ModelType.MEDIUM,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.rec_img_shape": [3, 48, 320],

                "Det.model_path": self.model_v6_server_det_onnx,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MEDIUM,
                "Det.ocr_version": OCRVersion.PPOCRV6,

                "Det.limit_side_len": 1280,
                "Det.limit_type": "max",
                "Det.box_thresh": 0.5,

                "EngineConfig.onnxruntime.use_cuda": True,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,

                "Rec.rec_keys_path": self.txt_path_v6
            }
        }

        self.save_dir = "/content/drive/MyDrive/RapidVideoOCR/outputs" if self.linux else "./outputs"
        # Options
        self.target_scale = 2
        self.padding = 10
        self.max_height_in_all_folder = 0

    @staticmethod
    def check_exist_folder(folder_dir):
        folder = folder_dir
        # Check folder is exist
        print("Folder exists:", os.path.exists(folder))
        print("Files in folder:", os.listdir(folder))

    def get_limit_side_len(self, max_height):
        batch_height = self.batch_size * (max_height + self.padding) if self.is_batch_rec else max_height
        # if batch_height < 320:
        #     limit_side_len = 480
        # elif batch_height < 480:
        #     limit_side_len = 640
        if self.is_batch_rec:
            if batch_height < 736:
                limit_side_len = 736
            elif batch_height < 960:
                limit_side_len = 960
            else:
                limit_side_len = 1024
        else:
            if batch_height < 150:
                limit_side_len = [320, 480, 640]
            elif batch_height < 240:
                limit_side_len = [480, 640, 736]
            else:
                limit_side_len = [640, 736, 832]
        # Multiples of 32
        # if limit_side_len % 32 != 0:
        #     limit_side_len = ((limit_side_len // 32) + 1) * 32
        return limit_side_len

    def get_limit_side_len_target_scale(self, max_height):
        size_image_with_padding = self.batch_size * (max_height + self.padding)
        limit_side_len = int(self.target_scale * (self.batch_size * (max_height + self.padding)))
        # Require multiples of 32
        if limit_side_len % 32 != 0:
            limit_side_len = ((limit_side_len // 32) + 1) * 32
        return limit_side_len

    def get_max_height_in_all_folder(self, height):
        # Check max height images in all folder
        if self.max_height_in_all_folder < height:
            self.max_height_in_all_folder = height

    def only_ocr(self, rgb_dir):
        for folder in glob.glob(os.path.join(rgb_dir, "*/")):
            if os.path.isdir(folder):
                folder_name = os.path.basename(os.path.normpath(folder))
                print(f"\nProcessing directory: {folder}\n")
                # outputs/a.srt  outputs/a.txt
                # extractor(folder, save_dir, save_name=folder_name)

                image_files = os.path.join(folder, os.listdir(folder)[0])
                # print(image_files)

                if not image_files:
                    print(f"No images found in {folder}")
                    continue

                with Image.open(image_files) as img:
                    w, h = img.size  # h = crop_height (giả sử crop sub dọc)

                # print(f"My height: {h}")
                if not self.linux:
                    self.get_max_height_in_all_folder(h)

                # if h >= 60:
                #     rec_img_shape = [3, 64, 320]
                # elif h >= 36:
                #     rec_img_shape = [3, 48, 320]
                # else:
                #     rec_img_shape = [3, 48, 320]

                rec_img_shape = [3, 48, 320]

                # With Det.limit_side_len / min
                folder_ocr_params = deepcopy(self.ocr_input_params["ocr_params"])
                folder_ocr_params["Rec.rec_img_shape"] = rec_img_shape
                folder_ocr_params["Det.limit_side_len"] = self.get_limit_side_len(max_height=h) if self.is_batch_rec else self.get_limit_side_len(max_height=h)[0]
                folder_ocr_params["Det.limit_type"] = "min"

                folder_ocr_params2 = deepcopy(self.ocr_input_params["ocr_params"])
                folder_ocr_params2["Rec.rec_img_shape"] = rec_img_shape
                folder_ocr_params2["Det.limit_side_len"] = self.get_limit_side_len(max_height=h) if self.is_batch_rec else self.get_limit_side_len(max_height=h)[1]
                folder_ocr_params2["Det.limit_type"] = "min"

                folder_ocr_params3 = deepcopy(self.ocr_input_params["ocr_params"])
                folder_ocr_params3["Rec.rec_img_shape"] = rec_img_shape
                folder_ocr_params3["Det.limit_side_len"] = self.get_limit_side_len(max_height=h) if self.is_batch_rec else self.get_limit_side_len(max_height=h)[2]
                folder_ocr_params3["Det.limit_type"] = "min"


                # With Det.limit_side_len / max
                folder_ocr_params4 = deepcopy(self.ocr_input_params["ocr_params"])
                folder_ocr_params4["Rec.rec_img_shape"] = rec_img_shape
                folder_ocr_params4["Det.limit_side_len"] = 1280
                folder_ocr_params4["Det.limit_type"] = "max"
                # folder_ocr_params["Global.max_side_len"] = w * target_scale
                # print(folder_ocr_params)
                # print(folder_ocr_params2)
                # print(folder_ocr_params3)


                # Version GPU for google colab
                folder_extractor = RapidVideOCR(
                    RapidVideOCRInput(
                        is_batch_rec=self.is_batch_rec,
                        batch_size=self.batch_size,
                        out_format="srt",
                        ocr_params_list=[folder_ocr_params, folder_ocr_params2, folder_ocr_params3, folder_ocr_params4]
                    )
                )
                folder_extractor(folder, self.save_dir, save_name=folder_name)


        if not self.linux:
            print(f"Max height image in all folder: {self.max_height_in_all_folder}px")


# Document: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/#__tabbed_3_4

def only_ocr_worker(args):
    rgb_dir, linux, is_batch_rec = args
    pid = os.getpid()
    print(f"[PID {pid}] Processing folder: {rgb_dir}")
    ocr = OCR(linux=linux, is_batch_rec=is_batch_rec, engine_type="paddle")  # Initialize each process separately
    return ocr.only_ocr(rgb_dir)


def main():
    linux = os.name == "posix"
    is_batch_rec = False
    # Folder containing subfolders containing images
    rgb_dir_list = [f"images/{i}" for i in range(1, 4)]

    # Unzip file
    for i in range(1, len(rgb_dir_list) + 1):
        # Đường dẫn folder gốc chứa zip file
        zip_dir = f'/content/drive/MyDrive/RapidVideoOCR/images/{i}' if linux else f'./test/{i}'
        # Đường dẫn folder đích
        out_dir = f'images/{i}'
        os.makedirs(out_dir, exist_ok=True)

        # Find all file zip in zip_dir
        zip_files = list(Path(zip_dir).glob('*.zip'))
        for zip_path in zip_files:
            print(f"Unzipping {zip_path} to {out_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(out_dir)

    # Create list args for each folder
    pool_args = [(rgb_dir, linux, is_batch_rec) for rgb_dir in rgb_dir_list]

    with Pool(processes=len(rgb_dir_list)) as pool:  # Adjust the number of processes to suit the GPU/CPU
        results = pool.map(only_ocr_worker, pool_args)
    # # Test
    # only_ocr_worker((rgb_dir_list[0], linux, is_batch_rec))


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()
# get cpu count
# import os
# print(os.cpu_count())

