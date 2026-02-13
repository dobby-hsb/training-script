import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt

CSV_INPUT = './dataset_metadata.csv'
PLOT_OUTPUT_HISTOGRAM = './likes_histogram.png'
PLOT_OUTPUT_RANGE = './likes_range_distribution.png'


def analyze_likes_statistics():
    """메타데이터 CSV에서 likes 통계를 분석"""
    likes_list = []
    feed_likes = {}  # feed_id별 likes (중복 제거용)

    with open(CSV_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feed_id = row['feed_id']
            likes = int(row['likes'])
            likes_list.append(likes)
            feed_likes[feed_id] = likes  # 동일 feed_id는 같은 likes 값

    # feed 단위 통계 (이미지가 여러 장인 feed도 1개로 계산)
    unique_likes = list(feed_likes.values())
    unique_likes_sorted = sorted(unique_likes)

    print("=" * 60)
    print("📊 Likes 통계 분석")
    print("=" * 60)

    # 기본 통계
    print(f"\n📁 전체 이미지 수: {len(likes_list):,}")
    print(f"📁 고유 피드 수: {len(unique_likes):,}")

    # 전체 이미지에 대한 Likes 통계
    print("\n" + "-" * 40)
    print("📈 전체 이미지 Likes 통계")
    print("-" * 40)

    total_all = sum(likes_list)
    count_all = len(likes_list)
    mean_all = total_all / count_all
    likes_sorted_all = sorted(likes_list)

    # 중앙값
    mid_all = count_all // 2
    if count_all % 2 == 0:
        median_all = (likes_sorted_all[mid_all - 1] + likes_sorted_all[mid_all]) / 2
    else:
        median_all = likes_sorted_all[mid_all]

    # 분위수
    q1_idx_all = count_all // 4
    q3_idx_all = (3 * count_all) // 4
    q1_all = likes_sorted_all[q1_idx_all]
    q3_all = likes_sorted_all[q3_idx_all]

    # 표준편차
    variance_all = sum((x - mean_all)**2 for x in likes_list) / count_all
    std_dev_all = variance_all**0.5

    print(f"  총합: {total_all:,}")
    print(f"  평균: {mean_all:.2f}")
    print(f"  중앙값: {median_all:.1f}")
    print(f"  표준편차: {std_dev_all:.2f}")
    print(f"  최소값: {min(likes_list):,}")
    print(f"  최대값: {max(likes_list):,}")
    print(f"  Q1 (25%): {q1_all}")
    print(f"  Q3 (75%): {q3_all}")

    print("\n" + "-" * 40)
    print("📈 피드 단위 Likes 통계")
    print("-" * 40)

    total = sum(unique_likes)
    count = len(unique_likes)
    mean = total / count

    # 중앙값
    mid = count // 2
    if count % 2 == 0:
        median = (unique_likes_sorted[mid - 1] + unique_likes_sorted[mid]) / 2
    else:
        median = unique_likes_sorted[mid]

    # 분위수
    q1_idx = count // 4
    q3_idx = (3 * count) // 4
    q1 = unique_likes_sorted[q1_idx]
    q3 = unique_likes_sorted[q3_idx]

    # 표준편차
    variance = sum((x - mean)**2 for x in unique_likes) / count
    std_dev = variance**0.5

    print(f"  총합: {total:,}")
    print(f"  평균: {mean:.2f}")
    print(f"  중앙값: {median:.1f}")
    print(f"  표준편차: {std_dev:.2f}")
    print(f"  최소값: {min(unique_likes):,}")
    print(f"  최대값: {max(unique_likes):,}")
    print(f"  Q1 (25%): {q1}")
    print(f"  Q3 (75%): {q3}")

    # 구간별 분포 - 피드 단위
    print("\n" + "-" * 40)
    print("📊 피드 단위 Likes 구간별 분포")
    print("-" * 40)

    ranges = [
        (10, 19, "10-19"),
        (20, 49, "20-49"),
        (50, 99, "50-99"),
        (100, 199, "100-199"),
        (200, 499, "200-499"),
        (500, 999, "500-999"),
        (1000, float('inf'), "1000+"),
    ]

    for low, high, label in ranges:
        count_in_range = sum(1 for x in unique_likes if low <= x <= high)
        percentage = (count_in_range / count) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {label:>8}: {count_in_range:>6,} ({percentage:>5.1f}%) {bar}")

    # 구간별 분포 - 전체 이미지 단위
    print("\n" + "-" * 40)
    print("📊 전체 이미지 Likes 구간별 분포")
    print("-" * 40)

    for low, high, label in ranges:
        count_in_range_all = sum(1 for x in likes_list if low <= x <= high)
        percentage_all = (count_in_range_all / count_all) * 100
        bar_all = "█" * int(percentage_all / 2)
        print(f"  {label:>8}: {count_in_range_all:>6,} ({percentage_all:>5.1f}%) {bar_all}")

    # 상위 10개 likes 값
    print("\n" + "-" * 40)
    print("🏆 Top 10 가장 많은 Likes (피드 단위)")
    print("-" * 40)

    top_feeds = sorted(feed_likes.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feed_id, likes) in enumerate(top_feeds, 1):
        print(f"  {i:>2}. {feed_id}: {likes:,} likes")

    # likes 값 빈도
    print("\n" + "-" * 40)
    print("📉 가장 흔한 Likes 값 (Top 10, 피드 단위)")
    print("-" * 40)

    likes_counter = Counter(unique_likes)
    for likes_val, freq in likes_counter.most_common(10):
        print(f"  {likes_val:>6} likes: {freq:>5,} 피드")

    # 다운로드 통계
    print("\n" + "-" * 40)
    print("⬇️ 다운로드 통계 (가장 높은 Likes부터)")
    print("-" * 40)

    likes_sorted_desc = sorted(likes_list, reverse=True)

    min_likes_downloaded = likes_sorted_desc[downloaded_count - 1]

    print(f"  최소 Likes: {min_likes_downloaded:,}")
    print(f"  최대 Likes: {max(likes_sorted_desc[:downloaded_count]):,}")

    # Likes 단위별 분포 (전체 이미지)
    print("\n" + "-" * 40)
    print("📊 전체 이미지 Likes 단위별 분포 (상위 100개)")
    print("-" * 40)

    likes_counter_all = Counter(likes_list)
    for likes_val, freq in likes_counter_all.most_common(100):
        percentage = (freq / count_all) * 100
        bar = "█" * int(percentage / 0.2)  # 0.2%마다 하나의 █
        print(f"  {likes_val:>6} likes: {freq:>6,} ({percentage:>5.1f}%) {bar}")

    print("\n" + "=" * 60)

    return unique_likes, likes_list


def plot_likes_distribution(unique_likes):
    """Likes 분포를 시각화"""
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. 히스토그램 (세밀한 분포)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(unique_likes, bins=50, edgecolor='white', alpha=0.8, color='#4C72B0')
    ax.set_xlabel('Likes', fontsize=12)
    ax.set_ylabel('Count (Feeds)', fontsize=12)
    ax.set_title('Likes Distribution Histogram', fontsize=14, fontweight='bold')
    mean_val = sum(unique_likes) / len(unique_likes)
    ax.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_HISTOGRAM, dpi=150, bbox_inches='tight')
    print(f"📊 히스토그램 저장됨: {PLOT_OUTPUT_HISTOGRAM}")
    plt.close()

    # 2. 구간별 바 차트
    fig, ax = plt.subplots(figsize=(10, 6))

    ranges = [
        (10, 14, "10-14"),
        (15, 19, "15-19"),
        (20, 24, "20-24"),
        (25, 29, "25-29"),
        (30, 39, "30-39"),
        (40, 49, "40-49"),
        (50, 74, "50-74"),
        (75, 99, "75-99"),
        (100, float('inf'), "100+"),
    ]

    labels = []
    counts = []

    for low, high, label in ranges:
        count = sum(1 for x in unique_likes if low <= x <= high)
        labels.append(label)
        counts.append(count)

    colors = plt.cm.Blues([0.3 + 0.07 * i for i in range(len(ranges))])
    bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.2)

    # 막대 위에 수치 표시
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 500,
                    f'{count:,}',
                    ha='center',
                    va='bottom',
                    fontsize=9)

    ax.set_xlabel('Likes Range', fontsize=12)
    ax.set_ylabel('Count (Feeds)', fontsize=12)
    ax.set_title('Feeds by Likes Range', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_RANGE, dpi=150, bbox_inches='tight')
    print(f"📊 구간별 분포 저장됨: {PLOT_OUTPUT_RANGE}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Likes 통계 분석 및 다운로드 이미지 수 계산')
    parser.add_argument('--min-likes', type=int, default=None, help='지정된 Likes 이상의 이미지 개수를 표시합니다 (가장 높은 Likes부터 계산)')

    args = parser.parse_args()

    unique_likes, likes_list = analyze_likes_statistics()
    plot_likes_distribution(unique_likes)

    # min_likes 옵션이 주어진 경우
    if args.min_likes is not None:
        likes_sorted_desc = sorted(likes_list, reverse=True)
        count_with_min_likes = sum(1 for x in likes_list if x >= args.min_likes)

        print("\n" + "=" * 60)
        print(f"🎯 Likes >= {args.min_likes:,} 통계")
        print("=" * 60)
        print(f"  이미지 개수: {count_with_min_likes:,}")
        print(f"  전체 중 비율: {(count_with_min_likes / len(likes_list)) * 100:.2f}%")

        if count_with_min_likes > 0:
            filtered_likes = [x for x in likes_list if x >= args.min_likes]
            min_val = min(filtered_likes)
            max_val = max(filtered_likes)
            mean_val = sum(filtered_likes) / len(filtered_likes)

            print(f"  최소 Likes: {min_val:,}")
            print(f"  최대 Likes: {max_val:,}")
            print(f"  평균 Likes: {mean_val:.2f}")

        print("=" * 60)
