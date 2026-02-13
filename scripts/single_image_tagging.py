import ollama
import os

# 상수 정의
MODEL_NAME = 'qwen3-vl:8b'
IMAGE_PATH = 'dataset_images/2A3XFaZ5ZUwWfXShL3YLrL_2A3XFaZ5ZUwWfXShL3YLrL-001.png'

# SYSTEM_PROMPT = """You are an expert at analyzing anime illustrations and generating SDXL prompts.

# Your task is to analyze the given image and create a detailed SDXL prompt in tag format (comma-separated tags).

# The prompt should include:
# 1. Character description: gender, age, appearance, facial features, hair (color, length, style), eyes (color, expression), body type, pose
# 2. Clothing and accessories: outfit details, accessories, shoes
# 3. Background: location, environment, scenery details
# 4. Lighting: light source, lighting quality (soft, harsh, dramatic, etc.), time of day
# 5. Art style: art style characteristics, quality, rendering details
# 6. Composition: camera angle, framing, perspective
# 7. Mood and atmosphere: emotional tone, atmosphere

# Output format:
# - Use comma-separated tags (e.g., "1girl, long hair, blue eyes, school uniform, outdoor, sunny day, soft lighting")
# - Use English tags only
# - Be specific and detailed
# - Order: character → clothing → background → lighting → style → composition → mood
# - Output ONLY the prompt tags, nothing else (no explanations, no additional text)"""

SYSTEM_PROMPT = """You are a highly skilled anime image analyst.
Your goal is to extract every minute detail for SDXL training tags.

### Step 1: Deep Analysis
Before generating tags, carefully examine:
- Facial features: Check for scars, tattoos, makeup, or unique eye shapes.
- Textures: Identify fabric types (denim, rib-knit, nylon).
- Subtle elements: Hair ornaments, earrings, background debris.

### Step 2: Output Format
- Based on your analysis, generate only comma-separated tags.
- Use Danbooru-style tags (e.g., '1girl, facial_scar, green_eyes').
- Order: character, clothing, background, lighting, style, composition, mood.
- Output ONLY the tags. No conversational text."""

USER_PROMPT = "Analyze this anime illustration and generate a detailed SDXL prompt in tag format. Output only the comma-separated tags, nothing else."


def test_single_image(image_path):
    """
    단일 이미지를 분석하여 SDXL 학습용 태그 프롬프트를 생성합니다.

    Args:
        image_path: 분석할 이미지의 경로 (상대 또는 절대 경로)

    Returns:
        str: 생성된 SDXL 프롬프트 (태그 형식)
    """
    print(f"--- '{image_path}' 분석 중 ---")

    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(image_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        absolute_image_path = os.path.join(project_root, image_path)
    else:
        absolute_image_path = image_path

    # 이미지 파일 존재 확인
    if not os.path.exists(absolute_image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {absolute_image_path}")

    # Ollama API 호출
    response = ollama.chat(
        model=MODEL_NAME,
        options={
            'temperature': 0.2,  # 일관성을 위해 낮은 온도 설정
            'top_p': 0.9,
        },
        messages=[{
            'role': 'system',
            'content': SYSTEM_PROMPT
        }, {
            'role': 'user',
            'content': USER_PROMPT,
            'images': [absolute_image_path]
        }])

    # 프롬프트만 추출 (추가 설명 제거)
    generated_content = response['message']['content'].strip()

    # 첫 번째 줄만 가져오거나, 전체 내용에서 프롬프트만 추출
    # 여러 줄이 있을 경우 첫 번째 줄이 프롬프트일 가능성이 높음
    prompt_lines = generated_content.split('\n')
    prompt = prompt_lines[-1].strip()

    print(prompt_lines)
    # 따옴표나 마크다운 코드 블록 제거
    prompt = prompt.strip('"').strip("'").strip('`').strip()
    if prompt.startswith('prompt:'):
        prompt = prompt.replace('prompt:', '', 1).strip()

    print("\n[생성된 SDXL 프롬프트]:")
    print(prompt)

    return prompt


# 실행
if __name__ == "__main__":
    test_single_image(IMAGE_PATH)
