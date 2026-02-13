import os
import csv
from urllib.parse import urlparse

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 데이터베이스 설정
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'dobby_canvas')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')

# AWS 설정
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_ACCESS_SECRET_KEY')
S3_BUCKET = os.environ.get('AWS_S3_BUCKET')

DOWNLOAD_DIR = './dataset_images'
CSV_OUTPUT = './dataset_metadata.csv'

# S3 클라이언트 생성 (명시적 자격 증명 사용)
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


def download_images_from_sql_results(sql_results):
    """
    SQL 쿼리 결과를 받아서 이미지를 다운로드하는 함수

    sql_results 예시:
    [
        {
            'feed_id': 'xxx',
            'url_prefix': 'https://cloudfront.../cartoon-feeds/2024-01-01/xxx',
            'filename_list': 'xxx-001.jpg,xxx-002.jpg',
            'likes': 15
        },
        ...
    ]
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    metadata = []

    for row in sql_results:
        feed_id = row['feed_id']
        url_prefix = row['url_prefix']
        filename_list = row['filename_list']
        likes = row['likes']

        # CloudFront URL에서 S3 경로 추출
        # url_prefix: https://cloudfront.../cartoon-feeds/2024-01-01/xxx
        # s3_prefix: cartoon-feeds/2024-01-01/xxx
        parsed_url = urlparse(url_prefix)
        s3_prefix = parsed_url.path.lstrip('/')

        # filename_list 파싱 (쉼표로 구분된 파일명들)
        filenames = [f.strip() for f in filename_list.split(',')]

        for filename in filenames:
            s3_key = f"{s3_prefix}/{filename}"
            local_filename = f"{feed_id}_{filename}"
            local_path = os.path.join(DOWNLOAD_DIR, local_filename)

            try:
                if not os.path.exists(local_path):
                    # S3에서 다운로드
                    s3_client.download_file(S3_BUCKET, s3_key, local_path)

                    # 메타데이터 저장
                    metadata.append({
                        'feed_id': feed_id,
                        'filename': local_filename,
                        's3_key': s3_key,
                        'likes': likes,
                        'url': f"{url_prefix}/{filename}"
                    })

                print(f"Downloaded: {local_filename} (likes: {likes})")

            except Exception as e:
                print(f"Error downloading {s3_key}: {e}")

    # CSV로 메타데이터 저장
    # if metadata:
    #     with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
    #         writer = csv.DictWriter(f, fieldnames=['feed_id', 'filename', 's3_key', 'likes', 'url'])
    #         writer.writeheader()
    #         writer.writerows(metadata)

    print(f"\nTotal images downloaded: {len(metadata)}")
    print(f"Metadata saved to: {CSV_OUTPUT}")


def fetch_rows(query):
    """PostgreSQL 데이터베이스에서 쿼리 실행 후 결과를 딕셔너리 리스트로 반환"""
    connection = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def main():
    query = """
SELECT
    cf.feed_id,
    cf.url_prefix,
    cf.filename_list,
    cfs.likes
FROM cartoon_feeds cf
INNER JOIN cartoon_feed_scores cfs ON cf.feed_id = cfs.feed_id
WHERE
    cfs.likes >= 20
    AND cf.deleted = false
ORDER BY cfs.likes DESC;
"""

    sql_results = fetch_rows(query)
    download_images_from_sql_results(sql_results)


if __name__ == '__main__':
    main()
