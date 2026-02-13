import argparse
import os
from pathlib import Path
from typing import List, Tuple

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

# 상수 정의
DEFAULT_IMAGE_DIR = "dataset_images"
DEFAULT_OUTPUT_CSV = "metadata.csv"

# 지원하는 이미지 확장자
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

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
    parser = argparse.ArgumentParser(
        description="WaifuDiffusion Tagger를 사용하여 폴더 내 모든 이미지에 태그를 생성하고 metadata.csv를 만듭니다.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help=f"태깅할 이미지가 있는 폴더 경로 (기본값: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"출력할 metadata.csv 파일 경로 (기본값: {DEFAULT_OUTPUT_CSV})",
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
    parser.add_argument(
        "--include-character",
        action="store_true",
        help="캐릭터 태그도 포함 (기본값: 일반 태그만 포함)",
    )
    parser.add_argument(
        "--filter-mode",
        action="store_true",
        help="dataset_metadata.csv에서 likes 필터링 모드 사용",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="dataset_metadata.csv",
        help="필터링에 사용할 메타데이터 CSV 파일 경로 (기본값: dataset_metadata.csv)",
    )
    parser.add_argument(
        "--min-likes",
        type=int,
        default=30,
        help="최소 likes 수 (기본값: 30)",
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


def find_image_files(image_dir: str) -> List[Path]:
    """
    폴더 내 모든 이미지 파일을 찾습니다.

    Args:
        image_dir: 이미지 폴더 경로

    Returns:
        List[Path]: 이미지 파일 경로 리스트
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(image_dir_path.glob(f"*{ext}"))
        image_files.extend(image_dir_path.glob(f"*{ext.upper()}"))

    # 파일명으로 정렬
    image_files.sort()
    return image_files


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

    def predict_tags(
        self,
        image,
        model_repo,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
        include_character=False,
    ) -> str:
        """
        이미지에 대한 태그를 예측하고 문자열로 반환합니다.

        Args:
            image: PIL Image 객체
            model_repo: 모델 저장소 이름
            general_thresh: 일반 태그 임계값
            general_mcut_enabled: 일반 태그 MCut 사용 여부
            character_thresh: 캐릭터 태그 임계값
            character_mcut_enabled: 캐릭터 태그 MCut 사용 여부
            include_character: 캐릭터 태그 포함 여부

        Returns:
            str: 태그 문자열 (쉼표로 구분)
        """
        self.load_model(model_repo)

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # 일반 태그: 예측 신뢰도 > 임계값인 것 선택
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # 일반 태그를 신뢰도 순으로 정렬
        sorted_general = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
        tag_list = [x[0] for x in sorted_general]

        # 캐릭터 태그 포함 여부에 따라 처리
        if include_character:
            character_names = [labels[i] for i in self.character_indexes]

            if character_mcut_enabled:
                character_probs = np.array([x[1] for x in character_names])
                character_thresh = mcut_threshold(character_probs)
                character_thresh = max(0.15, character_thresh)

            character_res = [x for x in character_names if x[1] > character_thresh]
            character_res = dict(character_res)

            # 캐릭터 태그를 신뢰도 순으로 정렬하여 추가
            sorted_character = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
            character_tags = [x[0] for x in sorted_character]
            tag_list.extend(character_tags)

        # 태그 문자열 생성 (괄호 이스케이프 처리)
        tag_string = ", ".join(tag_list).replace("(", r"\(").replace(")", r"\)")

        return tag_string


def get_absolute_path(path: str) -> str:
    """
    상대 경로를 절대 경로로 변환합니다.

    Args:
        path: 경로 (상대 또는 절대)

    Returns:
        str: 절대 경로
    """
    if os.path.isabs(path):
        return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, path)


def process_images(
    image_dir: str,
    output_csv: str,
    model_repo: str,
    general_threshold: float,
    general_mcut: bool,
    character_threshold: float,
    character_mcut: bool,
    include_character: bool,
) -> None:
    """
    폴더 내 모든 이미지를 태깅하고 metadata.csv를 생성합니다.

    Args:
        image_dir: 이미지 폴더 경로
        output_csv: 출력 CSV 파일 경로
        model_repo: 모델 저장소 이름
        general_threshold: 일반 태그 임계값
        general_mcut: 일반 태그 MCut 사용 여부
        character_threshold: 캐릭터 태그 임계값
        character_mcut: 캐릭터 태그 MCut 사용 여부
        include_character: 캐릭터 태그 포함 여부
    """
    # 경로 변환
    image_dir = get_absolute_path(image_dir)
    output_csv = get_absolute_path(output_csv)

    # 이미지 파일 찾기
    print(f"이미지 폴더: {image_dir}")
    image_files = find_image_files(image_dir)
    total_images = len(image_files)

    if total_images == 0:
        raise ValueError(f"이미지 파일을 찾을 수 없습니다: {image_dir}")

    print(f"총 {total_images}개의 이미지 파일을 찾았습니다.")
    print(f"모델: {model_repo}")
    print(f"일반 태그 임계값: {general_threshold}")
    print(f"캐릭터 태그 임계값: {character_threshold}")
    print(f"일반 태그 MCut: {general_mcut}")
    print(f"캐릭터 태그 MCut: {character_mcut}")
    print(f"캐릭터 태그 포함: {include_character}")
    print("-" * 80)

    # Tagger 초기화
    tagger = WaifuDiffusionTagger()

    # 결과 저장용 리스트
    metadata_rows = []

    # 각 이미지 처리
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{total_images}] 처리 중: {image_path.name}")

            # 이미지 로드
            image = Image.open(image_path)

            # 태그 예측
            tag_string = tagger.predict_tags(
                image,
                model_repo,
                general_threshold,
                general_mcut,
                character_threshold,
                character_mcut,
                include_character,
            )

            # 메타데이터에 추가 (파일명만 저장, 경로는 제외)
            metadata_rows.append({"file_name": image_path.name, "text": tag_string})

            print(f"  완료: {len(tag_string)} 문자")

        except Exception as e:
            print(f"  오류 발생: {e}")
            # 오류가 발생한 이미지는 건너뛰기
            continue

    # CSV 파일로 저장
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n{'=' * 80}")
        print(f"metadata.csv 저장 완료: {output_csv}")
        print(f"총 {len(metadata_rows)}개의 이미지 태깅 완료")
        print(f"{'=' * 80}")
    else:
        print("\n처리된 이미지가 없습니다.")


def process_filtered_images(
    metadata_csv: str,
    image_dir: str,
    output_csv: str,
    model_repo: str,
    general_threshold: float,
    general_mcut: bool,
    character_threshold: float,
    character_mcut: bool,
    include_character: bool,
    min_likes: int = 30,
) -> None:
    """
    dataset_metadata.csv에서 likes가 min_likes 이상인 이미지만 태깅하고 meta.csv를 생성합니다.

    Args:
        metadata_csv: 메타데이터 CSV 파일 경로 (feed_id, filename, s3_key, likes, url 컬럼 포함)
        image_dir: 이미지 폴더 경로
        output_csv: 출력 CSV 파일 경로
        model_repo: 모델 저장소 이름
        general_threshold: 일반 태그 임계값
        general_mcut: 일반 태그 MCut 사용 여부
        character_threshold: 캐릭터 태그 임계값
        character_mcut: 캐릭터 태그 MCut 사용 여부
        include_character: 캐릭터 태그 포함 여부
        min_likes: 최소 likes 수 (기본값: 30)
    """
    # 경로 변환
    metadata_csv = get_absolute_path(metadata_csv)
    image_dir = get_absolute_path(image_dir)
    output_csv = get_absolute_path(output_csv)

    # 메타데이터 CSV 읽기
    print(f"메타데이터 CSV 파일: {metadata_csv}")
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"메타데이터 CSV 파일을 찾을 수 없습니다: {metadata_csv}")

    metadata_df = pd.read_csv(metadata_csv)
    print(f"전체 데이터 수: {len(metadata_df)}개")

    # likes >= min_likes 필터링
    filtered_df = metadata_df[metadata_df["likes"] >= min_likes]
    print(f"likes >= {min_likes} 필터링 후 데이터 수: {len(filtered_df)}개")

    # filename 컬럼에서 고유한 파일 목록 추출
    target_filenames = set(filtered_df["filename"].unique())
    print(f"고유 파일 수: {len(target_filenames)}개")

    # 이미지 폴더에서 실제 존재하는 파일 확인
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")

    existing_files = []
    for filename in target_filenames:
        file_path = image_dir_path / filename
        if file_path.exists():
            existing_files.append(file_path)
        else:
            print(f"  경고: 파일이 존재하지 않습니다 - {filename}")

    # 파일명으로 정렬
    existing_files.sort()
    total_images = len(existing_files)

    if total_images == 0:
        raise ValueError(f"필터링된 이미지 파일이 없습니다.")

    print(f"처리할 이미지 수: {total_images}개")
    print(f"모델: {model_repo}")
    print(f"일반 태그 임계값: {general_threshold}")
    print(f"캐릭터 태그 임계값: {character_threshold}")
    print(f"일반 태그 MCut: {general_mcut}")
    print(f"캐릭터 태그 MCut: {character_mcut}")
    print(f"캐릭터 태그 포함: {include_character}")
    print("-" * 80)

    # Tagger 초기화
    tagger = WaifuDiffusionTagger()

    # 결과 저장용 리스트
    metadata_rows = []

    # 각 이미지 처리
    for idx, image_path in enumerate(existing_files, 1):
        try:
            print(f"[{idx}/{total_images}] 처리 중: {image_path.name}")

            # 이미지 로드
            image = Image.open(image_path)

            # 태그 예측
            tag_string = tagger.predict_tags(
                image,
                model_repo,
                general_threshold,
                general_mcut,
                character_threshold,
                character_mcut,
                include_character,
            )

            # 메타데이터에 추가 (파일명만 저장, 경로는 제외)
            metadata_rows.append({"file_name": image_path.name, "text": tag_string})

            print(f"  완료: {len(tag_string)} 문자")

        except Exception as e:
            print(f"  오류 발생: {e}")
            # 오류가 발생한 이미지는 건너뛰기
            continue

    # CSV 파일로 저장
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"\n{'=' * 80}")
        print(f"meta.csv 저장 완료: {output_csv}")
        print(f"총 {len(metadata_rows)}개의 이미지 태깅 완료")
        print(f"{'=' * 80}")
    else:
        print("\n처리된 이미지가 없습니다.")


def main():
    """메인 함수."""
    args = parse_args()

    if args.filter_mode:
        # 필터 모드: dataset_metadata.csv에서 likes로 필터링
        process_filtered_images(
            args.metadata_csv,
            args.image_dir,
            args.output_csv,
            args.model_repo,
            args.general_threshold,
            args.general_mcut,
            args.character_threshold,
            args.character_mcut,
            args.include_character,
            args.min_likes,
        )
    else:
        # 기본 모드: 폴더 내 모든 이미지 처리
        process_images(
            args.image_dir,
            args.output_csv,
            args.model_repo,
            args.general_threshold,
            args.general_mcut,
            args.character_threshold,
            args.character_mcut,
            args.include_character,
        )


if __name__ == "__main__":
    main()
