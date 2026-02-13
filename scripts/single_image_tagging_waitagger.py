import argparse
import os

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

# 상수 정의
IMAGE_PATH = 'dataset_images/2A3XFaZ5ZUwWfXShL3YLrL_2A3XFaZ5ZUwWfXShL3YLrL-001.png'

# Hugging Face 토큰 (환경 변수에서 가져오기)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Dataset v3 시리즈 모델
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Dataset v2 시리즈 모델
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# IdolSankaku 시리즈 모델
EVA02_LARGE_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-eva02-large-tagger-v1"
SWINV2_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-swinv2-tagger-v1"

# 기본 모델 (v3 시리즈 중 가장 일반적으로 사용되는 모델)
DEFAULT_MODEL_REPO = SWINV2_MODEL_DSV3_REPO

# 다운로드할 파일명
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# 기본 임계값
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
KAOMOJIS = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger를 사용하여 단일 이미지에 태그를 생성합니다.")
    parser.add_argument(
        "--image-path",
        type=str,
        default=IMAGE_PATH,
        help="태깅할 이미지 경로 (기본값: dataset_images/2A3XFaZ5ZUwWfXShL3YLrL_2A3XFaZ5ZUwWfXShL3YLrL-001.png)",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=DEFAULT_MODEL_REPO,
        help=f"사용할 모델 저장소 (기본값: {DEFAULT_MODEL_REPO})",
    )
    parser.add_argument(
        "--general-threshold",
        type=float,
        default=DEFAULT_GENERAL_THRESHOLD,
        help=f"일반 태그 임계값 (기본값: {DEFAULT_GENERAL_THRESHOLD})",
    )
    parser.add_argument(
        "--character-threshold",
        type=float,
        default=DEFAULT_CHARACTER_THRESHOLD,
        help=f"캐릭터 태그 임계값 (기본값: {DEFAULT_CHARACTER_THRESHOLD})",
    )
    parser.add_argument(
        "--general-mcut",
        action="store_true",
        help="일반 태그에 MCut 임계값 사용",
    )
    parser.add_argument(
        "--character-mcut",
        action="store_true",
        help="캐릭터 태그에 MCut 임계값 사용",
    )
    return parser.parse_args()


def load_labels(dataframe) -> tuple:
    """
    레이블 데이터프레임에서 태그 정보를 로드합니다.

    Args:
        dataframe: selected_tags.csv를 읽은 pandas DataFrame

    Returns:
        tuple: (tag_names, rating_indexes, general_indexes, character_indexes)
    """
    name_series = dataframe["name"]
    name_series = name_series.map(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x)
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).

    Args:
        probs: 확률 배열

    Returns:
        float: 계산된 임계값
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class WaifuDiffusionTagger:
    """WaifuDiffusion Tagger 모델을 사용하여 이미지에 태그를 생성하는 클래스."""

    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None

    def download_model(self, model_repo):
        """
        Hugging Face에서 모델과 레이블 파일을 다운로드합니다.

        Args:
            model_repo: 모델 저장소 이름

        Returns:
            tuple: (csv_path, model_path)
        """
        print(f"모델 다운로드 중: {model_repo}")
        csv_path = huggingface_hub.hf_hub_download(
            model_repo,
            LABEL_FILENAME,
            token=HF_TOKEN,
        )
        model_path = huggingface_hub.hf_hub_download(
            model_repo,
            MODEL_FILENAME,
            token=HF_TOKEN,
        )
        print("다운로드 완료!")
        return csv_path, model_path

    def load_model(self, model_repo):
        """
        모델을 로드합니다. 이미 로드된 모델이면 건너뜁니다.

        Args:
            model_repo: 모델 저장소 이름
        """
        if model_repo == self.last_loaded_repo and self.model is not None:
            return

        csv_path, model_path = self.download_model(model_repo)

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        if self.model is not None:
            del self.model
        model = rt.InferenceSession(model_path)
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.last_loaded_repo = model_repo
        self.model = model
        print(f"모델 로드 완료! (입력 크기: {self.model_target_size}x{self.model_target_size})")

    def prepare_image(self, image):
        """
        이미지를 모델 입력 형식으로 전처리합니다.

        Args:
            image: PIL Image 객체

        Returns:
            numpy.ndarray: 전처리된 이미지 배열
        """
        target_size = self.model_target_size

        # RGBA를 RGB로 변환 (이미지 모드에 따라 처리)
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        elif image.mode != "RGB":
            # 다른 모드(P, L, LA 등)를 RGB로 변환
            image = image.convert("RGB")

        # 이미지를 정사각형으로 패딩
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # 리사이즈
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # numpy 배열로 변환
        image_array = np.asarray(padded_image, dtype=np.float32)

        # PIL-native RGB를 BGR로 변환
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        """
        이미지에 대한 태그를 예측합니다.

        Args:
            image: PIL Image 객체
            model_repo: 모델 저장소 이름
            general_thresh: 일반 태그 임계값
            general_mcut_enabled: 일반 태그 MCut 사용 여부
            character_thresh: 캐릭터 태그 임계값
            character_mcut_enabled: 캐릭터 태그 MCut 사용 여부

        Returns:
            tuple: (sorted_general_strings, rating, character_res, general_res)
        """
        self.load_model(model_repo)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # 처음 4개 레이블은 실제로 rating: argmax로 하나 선택
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # 일반 태그: 예측 신뢰도 > 임계값인 것 선택
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)
            print(f"MCut으로 계산된 일반 태그 임계값: {general_thresh:.4f}")

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # 나머지는 캐릭터 태그: 예측 신뢰도 > 임계값인 것 선택
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)
            print(f"MCut으로 계산된 캐릭터 태그 임계값: {character_thresh:.4f}")

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        # 일반 태그를 신뢰도 순으로 정렬하여 문자열로 변환
        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (", ".join(sorted_general_strings).replace("(", r"\(").replace(")", r"\)"))

        return sorted_general_strings, rating, character_res, general_res


def get_absolute_path(image_path):
    """
    상대 경로를 절대 경로로 변환합니다.

    Args:
        image_path: 이미지 경로 (상대 또는 절대)

    Returns:
        str: 절대 경로
    """
    if os.path.isabs(image_path):
        return image_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, image_path)


def main():
    """메인 함수."""
    args = parse_args()

    # 이미지 경로 확인 및 변환
    image_path = get_absolute_path(args.image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    print(f"이미지 경로: {image_path}")
    print(f"모델: {args.model_repo}")
    print(f"일반 태그 임계값: {args.general_threshold}")
    print(f"캐릭터 태그 임계값: {args.character_threshold}")
    print(f"일반 태그 MCut: {args.general_mcut}")
    print(f"캐릭터 태그 MCut: {args.character_mcut}")
    print("-" * 80)

    # 이미지 로드
    image = Image.open(image_path)
    print(f"이미지 크기: {image.size}")

    # Tagger 초기화 및 예측
    tagger = WaifuDiffusionTagger()

    sorted_general_strings, rating, character_res, general_res = tagger.predict(
        image,
        args.model_repo,
        args.general_threshold,
        args.general_mcut,
        args.character_threshold,
        args.character_mcut,
    )

    # 결과 출력
    print("\n" + "=" * 80)
    print("태깅 결과")
    print("=" * 80)

    print("\n[Rating]")
    for key, value in sorted(rating.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {value:.4f}")

    print(f"\n[일반 태그] (임계값: {args.general_threshold})")
    print(f"  {sorted_general_strings}")
    print(f"\n  총 {len(general_res)}개 태그")

    print(f"\n[캐릭터 태그] (임계값: {args.character_threshold})")
    sorted_character = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
    character_strings = [f"{key} ({value:.4f})" for key, value in sorted_character]
    print(f"  {', '.join(character_strings)}")
    print(f"\n  총 {len(character_res)}개 태그")

    print("\n" + "=" * 80)
    print("\n[최종 태그 문자열]")
    print(sorted_general_strings)
    print("=" * 80)


if __name__ == "__main__":
    main()
