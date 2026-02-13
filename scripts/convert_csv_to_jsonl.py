"""CSV 파일 검증 및 수정 도구

metadata.csv 파일의 구조 문제를 찾아서 자동으로 수정합니다.
"""
import csv
import json
from pathlib import Path
from typing import List, Dict


def validate_csv(csv_path: str) -> None:
    """CSV 파일의 구조를 검증하고 문제를 출력합니다."""
    print(f"🔍 CSV 파일 검증 시작: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        # 첫 줄(헤더) 읽기
        header = f.readline().strip()
        expected_columns = len(header.split(','))
        print(f"📊 예상 컬럼 개수: {expected_columns}")
        print(f"📋 헤더: {header}")

        # 각 줄 검증
        line_num = 2  # 헤더 다음부터
        for line in f:
            actual_columns = len(line.split(','))
            if actual_columns != expected_columns:
                print(f"⚠️  라인 {line_num}: {actual_columns}개 컬럼 발견 (예상: {expected_columns})")
                print(f"   내용: {line[:100]}...")
            line_num += 1

    print(f"✅ 검증 완료: 총 {line_num - 1}개 라인")


def convert_csv_to_jsonl(csv_path: str, output_path: str) -> None:
    """CSV를 JSONL로 변환 (안전한 방식)"""
    print(f"\n🔄 CSV → JSONL 변환 시작")

    converted_count = 0
    error_count = 0

    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        # csv.DictReader는 쉼표가 포함된 텍스트도 올바르게 처리
        reader = csv.DictReader(csv_file)

        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for row_num, row in enumerate(reader, start=2):
                try:
                    # 필수 필드만 추출
                    json_obj = {"file_name": row.get("file_name", "").strip(), "text": row.get("text", "").strip()}

                    # 빈 값 체크
                    if not json_obj["file_name"] or not json_obj["text"]:
                        print(f"⚠️  라인 {row_num}: 빈 값 발견, 건너뜀")
                        error_count += 1
                        continue

                    jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    converted_count += 1

                except Exception as e:
                    print(f"❌ 라인 {row_num} 변환 실패: {e}")
                    error_count += 1

    print(f"✅ 변환 완료: {converted_count}개 성공, {error_count}개 실패")
    print(f"📁 저장 위치: {output_path}")


def main():
    """메인 실행 함수"""
    csv_path = "metadata.csv"
    output_path = "dataset_images/metadata.jsonl"

    # CSV 검증
    validate_csv(csv_path)

    # JSONL 변환
    convert_csv_to_jsonl(csv_path, output_path)

    print("\n🎉 모든 작업 완료!")
    print(f"\n📌 다음 단계:")
    print(f"1. {output_path} 파일이 생성되었는지 확인")
    print(f"2. run_script/train_lcm_distill_sdxl_lora.sh 실행")


if __name__ == "__main__":
    main()
