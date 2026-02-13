import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="WandB 결과 이미지(Target vs Online)를 비교하는 스크립트")
    parser.add_argument(
        "--wandb-root",
        type=str,
        default="results",
        help="WandB 로그가 저장된 최상위 폴더 경로 (기본값: wandb)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="비교 이미지가 저장될 폴더 경로 (기본값: comparison_results)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=20,
        help="이미지에 표시할 텍스트 크기",
    )
    return parser.parse_args()


def parse_filename(filename: str) -> Tuple[str, int, str]:
    """
    파일명에서 type, step, hash를 추출합니다.
    예: online_200_1b05dfc738caa5acd3e5.png -> ('online', 200, '1b05...')
    """
    # 패턴: (online 또는 target) _ (숫자) _ (나머지).png
    match = re.search(r"(online|target)_(\d+)_(.+)\.png", filename)
    if match:
        img_type, step, hash_val = match.groups()
        return img_type, int(step), hash_val
    return None, -1, ""


def process_run_folder(run_path: Path, output_root: Path, font_size: int) -> List[Dict[str, Any]]:
    """하나의 Run 폴더를 처리합니다."""
    # WandB는 설정에 따라 media/images/validation 또는 media/images에 바로 저장할 수 있음
    search_paths = [
        run_path / "files" / "media" / "images" / "validation",
        run_path / "files" / "media" / "images"
    ]

    all_pngs = []
    for img_dir in search_paths:
        if img_dir.exists():
            all_pngs.extend(list(img_dir.glob("*.png")))

    if not all_pngs:
        return []

    print(f"Processing run: {run_path.name}")

    # Step별로 이미지 분류
    steps_data = defaultdict(lambda: {"target": [], "online": []})

    for img_file in all_pngs:
        img_type, step, _ = parse_filename(img_file.name)
        if img_type in ["target", "online"]:
            steps_data[step][img_type].append(img_file)

    # 결과 저장 폴더 생성
    run_output_dir = output_root
    online_dir = run_output_dir / "online"
    target_dir = run_output_dir / "target"
    online_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    results_for_html = []

    # Step별 처리
    for step in sorted(steps_data.keys()):
        # 파일명(해시)이 다르므로 파일 수정 시간(mtime)으로 정렬하여 생성 순서를 맞춤
        targets = sorted(steps_data[step]["target"], key=lambda x: x.stat().st_mtime)
        onlines = sorted(steps_data[step]["online"], key=lambda x: x.stat().st_mtime)

        # 개수가 맞지 않아도 최대한 짝을 맞춰서 생성 (zip_longest 방식)
        max_len = max(len(targets), len(onlines))

        if len(targets) != len(onlines):
            print(f"  Warning: Step {step} has mismatched counts - Target: {len(targets)}, Online: {len(onlines)}")

        for i in range(max_len):
            t_path = targets[i] if i < len(targets) else None
            o_path = onlines[i] if i < len(onlines) else None

            save_name = f"step_{step:06d}_{i}.png"
            t_rel_path = ""
            o_rel_path = ""

            if t_path:
                dest = target_dir / save_name
                shutil.copy2(t_path, dest)
                t_rel_path = f"{run_path.name}/target/{save_name}"

            if o_path:
                dest = online_dir / save_name
                shutil.copy2(o_path, dest)
                o_rel_path = f"{run_path.name}/online/{save_name}"

            results_for_html.append({
                "step": step,
                "index": i,
                "target": t_rel_path,
                "online": o_rel_path
            })
            print(f"  Processed Step {step}, Index {i}")

    return results_for_html


def generate_html_report(output_dir: Path, all_results: Dict[str, List[Dict[str, Any]]]):
    """비교 결과를 한눈에 볼 수 있는 HTML 갤러리를 생성합니다."""
    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>WandB Comparison Gallery</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .run-container { background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 40px; overflow: hidden; }
            .run-header { background-color: #34495e; color: white; padding: 15px 25px; margin: 0; font-size: 1.2em; }
            .gallery { display: flex; flex-direction: column; gap: 20px; padding: 25px; }
            .comparison-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; background: #f9f9f9; padding: 15px; border-radius: 8px; }
            .image-box { text-align: center; }
            .image-box img { width: 100%; max-width: 512px; height: auto; border-radius: 4px; border: 1px solid #ddd; }
            .label { font-weight: bold; margin-bottom: 5px; display: block; color: #555; }
            .step-info { grid-column: span 2; font-weight: bold; border-bottom: 2px solid #34495e; padding-bottom: 5px; margin-bottom: 10px; }
            .no-image { background: #eee; display: flex; align-items: center; justify-content: center; height: 200px; color: #999; }
        </style>
    </head>
    <body>
        <h1>WandB Comparison Gallery (Target vs Online)</h1>
    """

    for run_name in sorted(all_results.keys()):
        if not all_results[run_name]:
            continue

        html_template += f'<div class="run-container"><h2 class="run-header">Run: {run_name}</h2><div class="gallery">'
        for item in all_results[run_name]:
            target_img = f'<img src="{item["target"]}" loading="lazy">' if item["target"] else '<div class="no-image">No Target Image</div>'
            online_img = f'<img src="{item["online"]}" loading="lazy">' if item["online"] else '<div class="no-image">No Online Image</div>'

            html_template += f"""
            <div class="comparison-row">
                <div class="step-info">Step: {item['step']} (Index: {item['index']})</div>
                <div class="image-box">
                    <span class="label">Target (Teacher)</span>
                    {target_img}
                </div>
                <div class="image-box">
                    <span class="label">Online (Student)</span>
                    {online_img}
                </div>
            </div>"""
        html_template += "</div></div>"

    html_template += "</body></html>"

    output_file = output_dir / "index.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\n✨ Gallery generated: {output_file}")


def main():
    args = parse_args()
    wandb_root = Path(args.wandb_root)
    output_dir = Path(args.output_dir)
    all_results = {}

    if not wandb_root.exists():
        print(f"Error: WandB root directory not found: {wandb_root}")
        return

    # wandb 폴더 내의 run-* 폴더들을 찾음
    for run_dir in wandb_root.glob("run-*"):
        if run_dir.is_dir():
            images = process_run_folder(run_dir, output_dir, args.font_size)
            if images:
                all_results[run_dir.name] = images

    if all_results:
        generate_html_report(output_dir, all_results)
    else:
        print("No images found to generate gallery.")


if __name__ == "__main__":
    main()