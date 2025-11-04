"""
데이터 전처리 및 파생변수 생성
"""
import re
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import BallTree

from src.utils.paths import load_data

EARTH_RADIUS_M = 6_371_000  # meters


def preprocess_data(train_df, test_df, cfg: DictConfig):
    """
    전체 전처리 파이프라인

    Args:
        train_df: train 데이터프레임
        test_df: test 데이터프레임
        cfg: Hydra 설정

    Returns:
        tuple: (train_df, valid_df, test_df, y_train, y_valid)
    """
    print("\n=== 데이터 전처리 시작 ===")

    # 0. 불필요한 컬럼 제거 (Feature Selection)
    print("0. Feature Selection 시작...")
    train_df, test_df = select_features(train_df, test_df, cfg)
    print(f"   After select_features: deposit in train? {cfg.data.target in train_df.columns}")

    # 1. 파생변수 생성
    train_df, test_df = create_features(train_df, test_df, cfg)
    print(f"   After create_features: deposit in train? {cfg.data.target in train_df.columns}")

    # 2. 결측치 처리
    print("\n2. 결측치 처리 시작...")
    train_df, test_df = handle_missing(train_df, test_df, cfg)
    print("   결측치 처리 완료")

    # 3. 학습용 데이터 준비 (수치형만 선택)
    print("\n3. 학습용 데이터 준비 시작...")
    train_df, test_df = prepare_for_training(train_df, test_df, cfg)
    print("   학습용 데이터 준비 완료")

    # 4. X, y 분리
    target_col = cfg.data.target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # 5. Train/Valid split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train,
        test_size=cfg.data.split.val_size,
        random_state=cfg.seed,
        shuffle=cfg.data.split.shuffle
    )

    print(f"\nTrain: {X_train.shape}, Valid: {X_valid.shape}, Test: {test_df.shape}")
    print("=== 전처리 완료 ===\n")

    return X_train, X_valid, test_df, y_train, y_valid


def select_features(train_df, test_df, cfg):
    """
    Feature Selection: 필수 컬럼만 남기고 불필요한 컬럼 제거

    제거 기준:
    1. 결측치 비율 90% 이상 컬럼 제거
    2. Target과 상관관계가 낮은 컬럼 제거 (상관계수 < 0.1)
    3. 다중공선성이 높은 컬럼 중 예측력 낮은 것 제거

    """
    

    # 필수 유지 컬럼 (target 포함)
    target_col = cfg.data.target
    keep_columns = [
        target_col,          # target (train만)
        '건축년도',
        '층',
        '전용면적(㎡)',
        '시군구',
        '계약년월',
        '도로명',
        '아파트명',
        '좌표X',
        '좌표Y',
        'k-건설사(시공사)',
        'k-전체동수',
        'k-연면적',
        '주차대수',
    ]

    # Train 데이터: target 포함
    train_keep = [col for col in keep_columns if col in train_df.columns]
    train_df = train_df[train_keep]

    # Test 데이터: target 제외
    test_keep = [col for col in keep_columns if col in test_df.columns and col != target_col]
    test_df = test_df[test_keep]

    print(f"   완료 - Train 컬럼 수: {len(train_df.columns)}, Test 컬럼 수: {len(test_df.columns)}")

    return train_df, test_df


def create_features(train_df, test_df, cfg):
    """파생변수 생성"""
    print("\n1. 파생변수 생성 중...")

    # bus, subway 데이터 로드 및 feature 추가
    if cfg.data.features.use_bus:
        print("   - 버스 feature 추가 중...")
        bus_df = load_data(cfg.data.files.bus_feature)
        train_df = add_bus_features(train_df, bus_df)
        test_df = add_bus_features(test_df, bus_df)
        print("   - 버스 feature 완료")

    if cfg.data.features.use_subway:
        print("   - 지하철 feature 추가 중...")
        subway_df = load_data(cfg.data.files.subway_feature)
        train_df = add_subway_features(train_df, subway_df)
        test_df = add_subway_features(test_df, subway_df)
        print("   - 지하철 feature 완료")

    # 기본 파생변수
    print("   - 기본 파생변수 생성 중 (아파트 등급, 건설사 등급 등)...")
    train_df = create_basic_features(train_df)
    test_df = create_basic_features(test_df)
    print("   - 기본 파생변수 생성 완료")

    print(f"   완료 - Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df



def _build_balltree(df_points: pd.DataFrame, lat_col: str, lon_col: str):
    """위경도 DataFrame으로부터 BallTree 생성 (haversine, 라디안 좌표)"""
    pts = df_points[[lat_col, lon_col]].dropna()
    if pts.empty:
        return None, None
    radians = np.radians(pts.to_numpy())
    tree = BallTree(radians, metric='haversine')
    return tree, pts.index  # 인덱스는 필요 시 사용

def _radius_flag(query_df: pd.DataFrame, tree: BallTree, lat_col: str, lon_col: str, radius_m: float) -> np.ndarray:
    """질의점들에 대해 반경 내 존재 여부 플래그 반환 (0/1)"""
    q = query_df[[lat_col, lon_col]].to_numpy()
    q_rad = np.radians(q)
    r_rad = radius_m / EARTH_RADIUS_M
    counts = tree.query_radius(q_rad, r=r_rad, count_only=True)
    return (counts > 0).astype(np.int8)


def add_bus_features(df, bus_df, radius_m: float = 500.0):
    """
    반경 radius_m(기본 500m) 내 버스정류장 존재 여부 '버스유무' 생성
    df: 좌표Y(위도), 좌표X(경도)
    bus_df: Y좌표(위도), X좌표(경도)
    """
    if '좌표X' not in df.columns or '좌표Y' not in df.columns:
        df['버스유무'] = np.int8(0)
        print("    좌표 정보 없음 - 버스유무 = 0")
        return df

    df['버스유무'] = np.int8(0)
    has_coords = df['좌표X'].notna() & df['좌표Y'].notna()
    apt_valid = df.loc[has_coords, ['좌표Y', '좌표X']].rename(columns={'좌표Y': 'lat', '좌표X': 'lon'})
    bus_valid = bus_df[['Y좌표', 'X좌표']].dropna().rename(columns={'Y좌표': 'lat', 'X좌표': 'lon'})

    if apt_valid.empty or bus_valid.empty:
        print("    유효 좌표 없음 - 버스유무 = 0")
        return df

    tree, _ = _build_balltree(bus_valid, 'lat', 'lon')
    flags = _radius_flag(apt_valid, tree, 'lat', 'lon', radius_m)
    df.loc[apt_valid.index, '버스유무'] = flags

    print(f"    완료 - 근처 버스 있음: {int(df['버스유무'].sum())}개")
    return df


def add_subway_features(df, subway_df, radius_m: float = 500.0):
    """
    반경 radius_m(기본 500m) 내 지하철역 존재 여부 '지하철유무' 생성
    df: 좌표Y(위도), 좌표X(경도)
    subway_df: 위도(lat), 경도(lon)
    """
    if '좌표X' not in df.columns or '좌표Y' not in df.columns:
        df['지하철유무'] = np.int8(0)
        print("    좌표 정보 없음 - 지하철유무 = 0")
        return df

    df['지하철유무'] = np.int8(0)
    has_coords = df['좌표X'].notna() & df['좌표Y'].notna()
    apt_valid = df.loc[has_coords, ['좌표Y', '좌표X']].rename(columns={'좌표Y': 'lat', '좌표X': 'lon'})
    sub_valid = subway_df[['위도', '경도']].dropna().rename(columns={'위도': 'lat', '경도': 'lon'})

    if apt_valid.empty or sub_valid.empty:
        print("    유효 좌표 없음 - 지하철유무 = 0")
        return df

    tree, _ = _build_balltree(sub_valid, 'lat', 'lon')
    flags = _radius_flag(apt_valid, tree, 'lat', 'lon', radius_m)
    df.loc[apt_valid.index, '지하철유무'] = flags

    print(f"    완료 - 근처 지하철 있음: {int(df['지하철유무'].sum())}개")
    return df


def create_basic_features(df):
    """기본 파생변수 생성"""
    # 계약년월 -> 년도/월 분리
    if '계약년월' in df.columns:
        df['계약년월'] = pd.to_datetime(df['계약년월'], format='%Y%m')

        df['계약년도'] = df['계약년월'].dt.year
        df['계약월'] = df['계약년월'].dt.month

        df.drop(columns=['계약년월'], inplace=True)

    # 시군구 -> 구/동 분리
    if  '시군구' in df.columns:
        df['구'] = df['시군구'].apply(lambda x: str(x).split(' ')[1] if len(str(x).split(' ')) > 1 else '기타')
        df['동'] = df['시군구'].apply(lambda x: str(x).split(' ')[2] if len(str(x).split(' ')) > 2 else '기타')

        df.drop(columns=['시군구'], axis=1, inplace=True)


    # 신축여부
    if '건축년도' in df.columns:
        df['신축여부'] = df['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)

    # 서울집값등급 파생변수
    if '구' in df.columns:
        price_group = {
            "강남구": 3, "서초구": 3, "송파구": 3, "용산구": 3,
            "마포구": 2, "성동구": 2, "광진구": 2, "동작구": 2, "양천구": 2, "강동구": 2,
        }
        df["서울집값등급"] = df["구"].map(price_group).fillna(1)

    if '아파트명' in df.columns:

        pattern_graded = [

            # ---- 5등급: 최상위 랜드마크/초고가 ----
            (r'한남더힐', 5),
            (r'나인원\s*한남|Nine\s*One\s*Hannam', 5),
            (r'PH\s*129|PH129', 5),
            (r'아크로\s*서울\s*포레스트|Acro\s*Seoul\s*Forest', 5),
            (r'아크로\s*리버\s*파크|Acro\s*River\s*Park', 5),

            # ---- 4등급: 강남권 최상위/프리미엄 플래그십 ----
            (r'원\s*베일리|원베일리|래미안\s*원\s*베일리', 4),
            (r'반포\s*자이', 4),
            (r'래미안\s*퍼스티지', 4),
            (r'압구정.*현대|현대.*압구정', 4),
            (r'디\s*에이치|THE\s*H|The\s*H', 4),             # 현대 고급 브랜드 (예: 디에이치 아너힐즈 등)
            (r'트리마제|Trimage', 4),
            (r'라클래시|래미안\s*라\s*클래시|Raemian\s*Laclass', 4),

            # ---- 3등급: 메이저 브랜드 일반/강남권 대단지 ----
            (r'래미안|Raemian', 3),
            (r'자이|Xi\b', 3),
            (r'아이\s*파크|I-?PARK|IPARK', 3),
            (r'푸르지오|Prugio', 3),
            (r'더\s*샵|The\s*Sharp', 3),
            (r'e\s*편한\s*세상|e편한세상|e-?Pyeonhansesang', 3),
            (r'센트레빌', 3),
            (r'래대푸|마포\s*래미안\s*푸르지오', 3),
            (r'헬리오\s*시티|Helio\s*City', 3),
            (r'파크리오', 3),
            (r'올림픽\s*선수촌', 3),
            (r'잠실\s*(리센츠|엘스|트리지움)', 3),

            # ---- 2등급: 준메이저/중상 브랜드, 대중적 신축/재건축 ----
            (r'힐스테이트|Hillstate', 2),
            (r'롯데\s*캐슬|Lotte\s*Castle', 2),
            (r'위브|We\'?ve', 2),
            (r'호반\s*(써밋|베르디움)', 2),
            (r'경남\s*아너스빌|아너스빌', 2),
            (r'두산', 2),
            (r'금호', 2),
            (r'벽산', 2),
            (r'현대\s*홈\s*타운', 2),

            # ---- 1등급: 기타/브랜드 미표기/중소 ----
            # 매칭 없으면 1로 처리
        ]
        def grade_score(name, pattern_graded):
            if pd.isna(name):
                return 1

            s = str(name).strip()

            for pattern, grade in pattern_graded:
                if re.search(pattern, s, flags=re.IGNORECASE):
                    return int(grade)
            return 1
        # 아파트명 기반 등급 부여
        df['아파트_등급'] = df['아파트명'].apply(lambda x: grade_score(x, pattern_graded)).astype(int)
        
        # 4,5 등급에 해당하는 아파트들은 집값 예측에 큰 영향을 미칠 것으로 판단해서
        # 별도 플래그를 통해 표시함.
        df['is_ultra_premium'] = df['아파트_등급'].isin([4,5]).astype(int)

        # value 값을 보니 1,2의 비중이 너무 큼 -> 3개 등급으로 재분류함
        bins_map_A = {1:1, 2:2, 3:3, 4:3, 5:3}
        df['아파트_등급'] = df['아파트_등급'].map(bins_map_A).astype('Int64')   
        
        df.drop(columns= ['아파트명'], inplace=True)


    # 건설사 등급 점수 매핑
    # 건설사도 집값에 큰 영향을 줄지는 모르지만 등급 별로 나누어 보는 것이 좋을 거라 판단해서
    # 파생변수 생성
    if 'k-건설사(시공사)' in df.columns:

        company_score_3 = {
            '삼성물산': 3,
            '현대건설': 3,
            'GS건설': 3,
            'DL이앤씨': 2,
            'HDC현대산업개발': 2,
            '대우건설': 2,
            '포스코건설': 2,
            '롯데건설': 2,
            '호반건설': 1,
            '금호건설': 1,
            '쌍용건설': 1,
            '두산건설': 1,
        }
        df['건설사등급'] = df['k-건설사(시공사)'].map(company_score_3).fillna(1).astype(int)

        df.drop(columns=['k-건설사(시공사)'], inplace=True)

    # 좋은_도로 파생변수
    if '도로명' in df.columns:
        # 프리미엄 도로 리스트
        premium_roads = [
            '강남대로', '테헤란로', '도산대로', '논현로', '압구정로', '봉은사로',
            '언주로', '반포대로', '서초대로', '잠실로', '올림픽로', '한강대로',
            '여의대로', '을지로', '종로', '세종대로', '퇴계로', '소공로', '명동길'
        ]

        # 주요 도로 리스트
        major_roads = [
            # 강남권
            '선릉로', '역삼로', '학동로', '신사역로', '압구정역로', '청담로', '삼성로',
            '영동대로', '선릉역로', '개포로', '일원로', '수서로',
            # 서초/반포
            '방배로', '사평대로', '동작대로', '현충로', '서래로',
            # 송파/잠실
            '백제고분로', '송파대로', '양재대로', '가락로', '문정로',
            # 여의도/마포
            '마포대로', '양화로', '서강대로', '홍익로', '합정로',
            # 강북 주요 도로
            '동호로', '장충단로', '충무로', '명륜길', '대학로', '성균관로',
            '창경궁로', '돈화문로', '인사동길', '삼일대로', '청계천로',
            # 용산/이태원
            '이태원로', '한남대로', '보광로', '소월로',
            # 영등포/구로
            '영등포로', '당산로', '선유로', '경인로',
            # 기타 주요 간선도로
            '강변북로', '내부순환로', '외곽순환고속도로', '경부고속도로'
        ]

        # 전체 좋은 도로 리스트
        all_good_roads = premium_roads + major_roads

        # 좋은_도로 피처 생성
        df['좋은_도로'] = df['도로명'].astype(str).str.contains(
            '|'.join(all_good_roads), na=False
        ).astype(int)

    return df


def handle_missing(train_df, test_df, cfg):
    """
    결측치 처리
    - 범주형 변수: "없음"으로 채우기
    - 수치형 변수: median/mean으로 채우기
    """
    if not cfg.data.preprocessing.handle_missing:
        return train_df, test_df

    print("\n2. 결측치 처리 중...")
    strategy = cfg.data.preprocessing.missing_strategy

    # 1. 범주형 변수 처리 (문자열, object 타입)
    categorical_cols = train_df.select_dtypes(include=['object', 'string']).columns

    for col in categorical_cols:
        if train_df[col].isnull().sum() > 0:
            train_df[col].fillna('없음', inplace=True)
            print(f"   - {col}: 결측치를 '없음'으로 채움")

        if col in test_df.columns and test_df[col].isnull().sum() > 0:
            test_df[col].fillna('없음', inplace=True)

    # 2. 수치형 변수 처리
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if train_df[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = train_df[col].median()
            elif strategy == 'mean':
                fill_value = train_df[col].mean()
            else:
                fill_value = 0  # default

            train_df[col].fillna(fill_value, inplace=True)
            print(f"   - {col}: 결측치를 {strategy}({fill_value:.2f})로 채움")

            if col in test_df.columns:
                test_df[col].fillna(fill_value, inplace=True)

    print("   완료")
    return train_df, test_df


def prepare_for_training(train_df, test_df, cfg):
    """
    학습용 데이터 준비:
    1. 범주형 변수 인코딩
    2. Log 변환 (특정 컬럼)
    """
    print("\n3. 학습용 데이터 준비 중...")

    # 1. 범주형 변수 인코딩
    train_df, test_df = encode_categorical(train_df, test_df)

    # 2. Log 변환
    train_df, test_df = apply_scaling(train_df, test_df)

    # 3. 수치형만 선택 (이미 인코딩되어 모두 수치형이지만, 혹시 모를 문자열 제거)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # target 컬럼 유지 (train에만 존재)
    target_col = cfg.data.target
    if target_col not in numeric_cols and target_col in train_df.columns:
        numeric_cols.append(target_col)

    train_df = train_df[numeric_cols]
    test_numeric_cols = [col for col in numeric_cols if col in test_df.columns and col != target_col]
    test_df = test_df[test_numeric_cols]

    print(f"   완료 - 최종 feature 개수: {len(train_df.columns)-1 if target_col in train_df.columns else len(train_df.columns)}")
    return train_df, test_df


def encode_categorical(train_df, test_df):
    """
    범주형 변수 인코딩
    - 구: 주요 구는 원핫인코딩, 나머지는 기타로 묶음
    - 동: 라벨 인코딩
    """
    print("\n   3-1. 범주형 인코딩 중...")

    # 1. 구 처리 (원핫인코딩)
    if '구' in train_df.columns:
        # 주요 구 리스트 (강남 3구 + 기타 주요 구)
        major_gu = ['강남구', '서초구', '송파구', '용산구', '마포구', '성동구', '광진구', '동작구', '양천구', '강동구']

        # 기타 구는 '기타구'로 통합
        train_df['구_grouped'] = train_df['구'].apply(lambda x: x if x in major_gu else '기타구')
        test_df['구_grouped'] = test_df['구'].apply(lambda x: x if x in major_gu else '기타구')

        # 원핫인코딩
        train_gu_dummies = pd.get_dummies(train_df['구_grouped'], prefix='구', drop_first=False)
        test_gu_dummies = pd.get_dummies(test_df['구_grouped'], prefix='구', drop_first=False)

        # test에 없는 컬럼은 0으로 추가
        for col in train_gu_dummies.columns:
            if col not in test_gu_dummies.columns:
                test_gu_dummies[col] = 0

        # train에 없는 컬럼 제거 (test에만 있는 경우)
        test_gu_dummies = test_gu_dummies[train_gu_dummies.columns]

        # 원본 컬럼 제거 후 원핫인코딩 추가
        train_df = train_df.drop(columns=['구', '구_grouped'])
        test_df = test_df.drop(columns=['구', '구_grouped'])
        train_df = pd.concat([train_df, train_gu_dummies], axis=1)
        test_df = pd.concat([test_df, test_gu_dummies], axis=1)

        print(f"      - 구: 원핫인코딩 완료 ({len(train_gu_dummies.columns)}개 컬럼)")

    # 2. 동 처리 (라벨 인코딩)
    if '동' in train_df.columns:
        le_dong = LabelEncoder()

        # train + test 합쳐서 fit (모든 카테고리 학습)
        all_dong = pd.concat([train_df['동'], test_df['동']], axis=0).astype(str)
        le_dong.fit(all_dong)

        train_df['동_encoded'] = le_dong.transform(train_df['동'].astype(str))
        test_df['동_encoded'] = le_dong.transform(test_df['동'].astype(str))

        train_df = train_df.drop(columns=['동'])
        test_df = test_df.drop(columns=['동'])

        print(f"      - 동: 라벨 인코딩 완료 (unique: {len(le_dong.classes_)})")

    print("      완료")
    return train_df, test_df


def apply_scaling(train_df, test_df):
    """
    로그 변환 적용:
    - 층, k-전체동수, k-연면적, 전용면적(㎡)에 로그 변환
    - Tree-based 모델이므로 표준화는 생략
    """
    print("\n   3-2. 로그 변환 중...")

    # Log 변환할 컬럼들
    log_cols = ['층', 'k-전체동수', 'k-연면적', '전용면적(㎡)']

    # 실제 존재하는 컬럼만 필터링
    log_cols_exist = [col for col in log_cols if col in train_df.columns]

    # Log 변환 (0보다 큰 값에만 적용)
    for col in log_cols_exist:
        # 0 이하 값은 1로 치환 (log(0) 방지)
        train_df[col] = train_df[col].apply(lambda x: x if x > 0 else 1)
        test_df[col] = test_df[col].apply(lambda x: x if x > 0 else 1)

        # Log 변환
        train_df[col] = np.log1p(train_df[col])
        test_df[col] = np.log1p(test_df[col])

    print(f"      - Log 변환 완료: {log_cols_exist}")
    print("      완료")

    return train_df, test_df
