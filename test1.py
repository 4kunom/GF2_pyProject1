import easyocr
import numpy as np
from PIL import Image, ImageDraw
import os
import pandas as pd
import glob

# 설정 템플릿
class Config:
    # 게임 해상도 (1920x1080)
    GAME_WIDTH, GAME_HEIGHT = 2560, 1440
    # 닉네임, 공적치, 누적공적치 각각에 맞는 박스 오프셋 (비율로 정의)
    START_Y = 0.2  # 시작 Y 값 (비율)
    STEP_Y = 0.118  # Y 값 간격 (비율)
    
    # 자동으로 오프셋 계산
    def generate_offsets(self, item, num_boxes=6):
        offsets = []
        for i in range(num_boxes):
            y_offset = self.START_Y + i * self.STEP_Y
            if item == "nickname":
                x_offset = 0.13
            elif item == "merit":
                x_offset = 0.53
            elif item == "cumulative_merit":
                x_offset = 0.65
            offsets.append((x_offset, y_offset))
        return offsets

    BBOX_OFFSETS = {
        "nickname": [],
        "merit": [],
        "cumulative_merit": []
    }

    # 항목별 자동 오프셋 생성
    def __init__(self):
        self.BBOX_OFFSETS["nickname"] = self.generate_offsets("nickname")
        self.BBOX_OFFSETS["merit"] = self.generate_offsets("merit")
        self.BBOX_OFFSETS["cumulative_merit"] = self.generate_offsets("cumulative_merit")
    
    # 각 박스의 크기 계산 비율 (게임 해상도를 기준으로 설정)
    BOX_SIZE_RATIOS = {
        "nickname": (0.2, 0.05),  # 닉네임 박스 크기 비율
        "merit": (0.1, 0.07),     # 공적치 박스 크기 비율
        "cumulative_merit": (0.1, 0.07)  # 누적 공적치 박스 크기 비율
    }
    # 이미지 폴더 경로
    IMAGE_FOLDER = r'G:\newTest\screenshots'
    # 결과 엑셀 파일 저장 경로
    EXCEL_OUTPUT_PATH = r'G:\newTest\screenshots\extracted_texts.xlsx'


# 텍스트 추출 함수
def calculate_bbox(image_size, game_size, offset_ratio, box_size_ratio):
    """
    해상도에 맞춰서 영역을 조정하고, 위치와 크기를 반환
    :param image_size: 원본 이미지의 크기 (width, height)
    :param game_size: 게임 화면 해상도 (width, height)
    :param offset_ratio: 영역 조정 비율 (좌/우, 상/하)
    :param box_size_ratio: 박스 크기 비율 (가로 비율, 세로 비율)
    :return: 조정된 영역 (left, top, right, bottom)
    """
    width, height = image_size

    # 박스 크기를 게임 해상도를 기준으로 계산
    box_width = int(game_size[0] * box_size_ratio[0])
    box_height = int(game_size[1] * box_size_ratio[1])

    # 화면 비율에 따라 오프셋 계산
    offset_x = int(offset_ratio[0] * game_size[0])
    offset_y = int(offset_ratio[1] * game_size[1])

    # 중앙에 맞춰서 좌표 계산
    left = (width - game_size[0]) // 2 + offset_x
    top = (height - game_size[1]) // 2 + offset_y

    return max(0, left), max(0, top), min(width, left + box_width), min(height, top + box_height)


# 텍스트 추출 함수
def extract_text(image_path, bboxes_offsets, box_size_ratios, game_size):
    """
    이미지에서 여러 영역을 추출하여 텍스트를 반환
    :param image_path: 이미지 파일 경로
    :param bboxes_offsets: 텍스트 추출할 영역의 오프셋
    :param box_size_ratios: 각 박스 크기 비율
    :param game_size: 게임 화면 해상도
    :return: 추출된 텍스트 리스트
    """
    reader = easyocr.Reader(['en', 'ko'])  # OCR 리더 초기화
    image = Image.open(image_path)  # 이미지 열기
    width, height = image.size
    draw = ImageDraw.Draw(image)  # 이미지 위에 사각형 그리기

    all_text = []  # 추출된 텍스트를 저장할 리스트
    for key, offsets in bboxes_offsets.items():
        box_size_ratio = box_size_ratios[key]  # 각 항목에 맞는 비율 적용
        for offset_ratio in offsets:
            # 추출할 영역을 조정하여 계산
            bbox = calculate_bbox((width, height), game_size, offset_ratio, box_size_ratio)
            cropped = image.crop(bbox)  # 영역을 잘라냄
            result = reader.readtext(np.array(cropped), detail=0)  # OCR로 텍스트 추출
            all_text.append((key, result))  # 항목 이름과 추출된 텍스트 저장
            draw.rectangle(bbox, outline="red", width=2)  # 추출된 영역을 빨간색 사각형으로 표시

    # 추출된 이미지를 저장
    image.save(f"{os.path.splitext(image_path)[0]}_output.png")
    return all_text


# 텍스트를 정규화하여 3개 항목으로 반환하는 함수
def normalize_row(extracted_data):
    """
    텍스트 배열을 정규화하여 3개 항목으로 반환
    :param extracted_data: OCR로 추출된 데이터 (각 항목에 대한 텍스트)
    :return: 정규화된 텍스트 배열 (항목 수가 3개로 맞춰짐)
    """
    # 항목 별로 데이터 추출
    nickname_list = []
    merit_list = []
    cumulative_merit_list = []
    
    for key, texts in extracted_data:
        if key == "nickname":
            nickname_list.extend(texts)  # 닉네임을 리스트로 추가
        elif key == "merit":
            merit_list.extend(texts)  # 공적치 리스트에 추가
        elif key == "cumulative_merit":
            cumulative_merit_list.extend(texts)  # 누적 공적치 리스트에 추가

    # 각 항목 리스트의 길이를 맞추기 위해 번호 추가
    max_len = max(len(nickname_list), len(merit_list), len(cumulative_merit_list))
    while len(nickname_list) < max_len:
        nickname_list.append(f"user{len(nickname_list)+1}")
    while len(merit_list) < max_len:
        merit_list.append("0")  # 빈 값 대신 0 추가
    while len(cumulative_merit_list) < max_len:
        cumulative_merit_list.append("0")  # 빈 값 대신 0 추가
    
    # 3개 항목으로 나누어 반환
    return list(zip(nickname_list, merit_list, cumulative_merit_list))


# 이미지 폴더 내의 모든 이미지 처리 함수
def process_images(image_folder, bboxes_offsets, box_size_ratios, game_size):
    """
    폴더 내 이미지들을 처리하고, 텍스트를 추출하여 리스트로 반환
    :param image_folder: 이미지 폴더 경로
    :param bboxes_offsets: 텍스트 추출할 영역의 오프셋
    :param game_size: 게임 화면 해상도
    :param box_size_ratios: 각 박스 크기 비율
    :return: 추출된 텍스트가 포함된 리스트
    """
    # 폴더 내 모든 이미지 파일 탐색
    files = glob.glob(os.path.join(image_folder, '*.png'))
    all_data = []  # 추출된 모든 데이터를 저장할 리스트

    for file in files:
        extracted_data = extract_text(file, bboxes_offsets, box_size_ratios, game_size)
        # 추출된 텍스트를 정규화하여 리스트에 추가
        normalized_text = normalize_row(extracted_data)
        all_data.extend(normalized_text)  # 여러 데이터를 하나의 리스트에 추가

    return all_data


# 엑셀 파일로 저장하는 함수
def save_to_excel(data, output_path):
    """
    추출된 데이터를 엑셀 파일로 저장
    :param data: 엑셀에 저장할 데이터 (리스트 형태)
    :param output_path: 엑셀 파일 저장 경로
    """
    df = pd.DataFrame(data, columns=["Nickname", "Merit", "Cumulative Merit"])
    df.to_excel(output_path, index=False)


# 실행 부분
if __name__ == "__main__":
    # 설정 클래스에서 템플릿 적용
    config = Config()  # 설정 객체 생성
    image_folder = config.IMAGE_FOLDER
    bboxes_offsets = config.BBOX_OFFSETS
    box_size_ratios = config.BOX_SIZE_RATIOS
    game_size = (config.GAME_WIDTH, config.GAME_HEIGHT)
    output_path = config.EXCEL_OUTPUT_PATH

    # 이미지 처리 및 엑셀 저장
    all_data = process_images(image_folder, bboxes_offsets, box_size_ratios, game_size)
    save_to_excel(all_data, output_path)

    # 확인을 위해 추출된 텍스트 출력
    for row in all_data:
        print(row)
