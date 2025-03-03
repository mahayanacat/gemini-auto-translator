import os
import threading
import time
import random
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from docx import Document
import requests
import logging
import subprocess
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
import gc
import uvicorn
import webbrowser
import json
import PyPDF2
import docx2txt
import codecs
import tiktoken
import signal
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_dir = os.path.join("D:/gemini/gemini-translated", "logs")
os.makedirs(log_dir, exist_ok=True)
app = FastAPI()

# 速率限制配置
REQUESTS_PER_MINUTE = 10
TIME_WINDOW = 60
BUCKET_CAPACITY = 10
REFILL_RATE = 1
TOKENS = BUCKET_CAPACITY
LAST_REFILL = time.time()

MAX_WORKERS = 1
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 全局变量
is_translating = True
is_thread_running = False
global_file_path = ""
global_style = ""
global_temperature = ""
global_batch_size = 10  # 默认翻译批次大小
global_batch_size_check = 5  # 默认校对批次大小
PROXY = {"http": "", "https": ""}
current_log_file = None  # 当前任务的日志文件路径
OUTPUT_DIR = "D:/gemini/gemini-translated/"
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

IS_UPLOADING_FLAG = os.path.join(TEMP_DIR, "is_uploading.flag")
IS_CHECKING_FLAG = os.path.join(TEMP_DIR, "is_checking.flag")

consecutive_api_failures = 0
MAX_CONSECUTIVE_API_FAILURES = 5
MAX_BATCH_RETRIES = 5  # 每批次最大重试次数

# Tesseract 和 Poppler 路径
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler\Library\bin"

# 验证工具安装
if not os.path.exists(TESSERACT_PATH):
    logging.error(f"Tesseract 未正确安装或路径错误：{TESSERACT_PATH}")
else:
    try:
        result = subprocess.run([TESSERACT_PATH, '--version'], capture_output=True, text=True, check=True)
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        logging.info(f"Tesseract 路径有效：{TESSERACT_PATH}, 版本: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Tesseract 可执行文件无法运行：{e.stderr}")
    except Exception as e:
        logging.error(f"Tesseract 测试失败：{str(e)}")

if not os.path.exists(POPPLER_PATH) or not os.path.isfile(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
    logging.error(f"Poppler 未正确安装或路径错误：{POPPLER_PATH}")
else:
    logging.info(f"Poppler 路径有效：{POPPLER_PATH}")

def token_bucket(func):
    def wrapper(*args, **kwargs):
        global TOKENS, LAST_REFILL
        now = time.time()
        time_since_last_refill = now - LAST_REFILL
        TOKENS = min(BUCKET_CAPACITY, TOKENS + time_since_last_refill * REFILL_RATE)
        LAST_REFILL = now
        if TOKENS >= 1:
            TOKENS -= 1
            return func(*args, **kwargs)
        else:
            wait_time = (1 - TOKENS) / REFILL_RATE
            logging.warning(f"令牌桶为空，等待 {wait_time:.2f} 秒...")
            time.sleep(wait_time)
            time.sleep(random.uniform(0.05, 0.1))
            return wrapper(*args, **kwargs)
    return wrapper

@token_bucket
def call_gemini_api(data, headers, model_url, api_key, max_retries=5):
    global consecutive_api_failures, is_translating
    full_url = f"{model_url}?key={api_key}"
    
    for attempt in range(max_retries):
        if not is_translating:
            logging.info("翻译任务已暂停，中断 API 请求")
            return None
        response = None
        try:
            response = requests.post(full_url, json=data, headers=headers, proxies=PROXY, timeout=300)
            response.raise_for_status()
            consecutive_api_failures = 0
            return response
        except requests.exceptions.RequestException as e:
            status_code = getattr(response, 'status_code', None) if response else None
            logging.error(f"API 请求失败 (Attempt {attempt + 1}/{max_retries}): {e}, Status Code: {status_code}, URL: {full_url}")
            consecutive_api_failures += 1
            if consecutive_api_failures >= MAX_CONSECUTIVE_API_FAILURES:
                logging.error(f"达到最大连续 API 请求失败次数 ({MAX_CONSECUTIVE_API_FAILURES})，自动暂停翻译任务！")
                is_translating = False
                return None
            if status_code == 429:
                wait_time = (3 ** attempt) + random.random()
                logging.warning(f"达到速率限制，等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            elif isinstance(e, requests.exceptions.ConnectionError):
                wait_time = (3 ** attempt) + random.random()
                logging.warning(f"连接错误，等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            else:
                logging.error(f"API 请求失败: {e}")
                return None
    logging.error("达到最大重试次数，API 请求失败")
    return None

def preprocess_image(image):
    """预处理图像以提高 OCR 准确性"""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)

def read_document(file_path):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".docx":
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif file_extension == ".doc":
            return docx2txt.process(file_path)
        elif file_extension == ".pdf":
            try:
                # 尝试直接提取文本（适用于可搜索 PDF）
                text = ""
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        extracted = reader.pages[page_num].extract_text()
                        if extracted.strip():
                            text += extracted + "\n"
                        else:
                            raise ValueError("可能是扫描件，切换到 OCR")
                return text
            except:
                logging.info(f"{file_path} 检测为扫描件，使用 OCR 处理")
                try:
                    # 使用 OCR 处理扫描件 PDF
                    images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
                    text = ""
                    for img in images:
                        processed_img = preprocess_image(img)
                        text += pytesseract.image_to_string(processed_img, lang='chi_sim', config='--psm 6') + "\n"
                    return text
                except Exception as e:
                    logging.error(f"OCR 处理失败：{str(e)}")
                    raise Exception(f"OCR 处理失败：{str(e)}")
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return file.read()
        else:
            logging.error(f"不支持的文件类型: {file_extension}")
            return None
    except FileNotFoundError:
        logging.error(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        logging.error(f"读取文件失败: {e}")
        return None

def clean_xml_string(text):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

class SpecificLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "补译句子" in msg or "长度差异过大" in msg

def setup_specific_logging(filename):
    global current_log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{filename}_specific_{timestamp}.log")
    current_log_file = log_file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(SpecificLogFilter())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return file_handler

def remove_specific_logging(handler):
    logging.getLogger().removeHandler(handler)

def log_translation_check(filename, sentence_index, orig_text, trans_text, issue=None, fixed=False, new_text=None):
    """记录翻译检查的回执"""
    if issue:
        if fixed and new_text:
            logging.info(f"句子 {sentence_index} 检查发现问题: {issue}，已修正")
        else:
            logging.warning(f"句子 {sentence_index} 检查发现问题: {issue}，未能修正")

def retranslate_sentence(sentence, style, api_key, model_url):
    """重译单句"""
    prompt = f"""请将以下文本翻译成流畅的简体字中文，并严格禁止拆分任何段落。完整翻译，禁止节译。只翻译，不解释。若遇古典诗句、颂文，按照严格工整的古诗格式翻译。禁止编造原文没有的内容。正确使用标点符号。每句译文前保留原文句子编号（如 [1]），只返回翻译结果。翻译风格：{style}。
    {sentence}"""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = call_gemini_api(data, headers, model_url, api_key)
    if response and "candidates" in response.json() and response.json()["candidates"]:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return None

def check_translation(original_text, chunk, style, filename, api_key, model_url):
    logging.info("正在校对...")
    batch_pairs = chunk.splitlines()
    corrected_lines = []

    for pair in batch_pairs:
        if "原文:" in pair and "译文:" in pair:
            orig_line = pair.split("原文:")[1].split("译文:")[0].strip()
            trans_line = pair.split("译文:")[1].strip()
            match = re.match(r'\[(\d+)\]', trans_line)
            if not match:
                continue
            sentence_index = int(match.group(1))

            # 检查 1: 译文是否完全是原文的直接输出
            if trans_line.strip() == orig_line.strip():
                issue = "译文完全是原文的直接输出"
                new_text = retranslate_sentence(orig_line, style, api_key, model_url)
                if new_text and new_text != orig_line:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, True, new_text)
                    corrected_lines.append(f"[{sentence_index}][改] {new_text}")
                else:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, False)
                    corrected_lines.append(f"[{sentence_index}][改] {trans_line}")
                continue

            # 检查 2: 译文是否准确传达原文意思（简单检查长度差异）
            orig_length = len(orig_line)
            trans_length = len(re.sub(r'\[\d+\]\s*', '', trans_line))
            if trans_length < orig_length / 2 or trans_length > orig_length * 2:
                issue = "译文缺失，标记为[未翻译]"
                new_text = retranslate_sentence(orig_line, style, api_key, model_url)
                if new_text and len(new_text) > orig_length / 2 and len(new_text) < orig_length * 2:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, True, new_text)
                    corrected_lines.append(f"[{sentence_index}][改] {new_text}")
                else:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, False)
                    corrected_lines.append(f"[{sentence_index}][改] {trans_line}")
                continue

            # 检查 3: 译文是否夹杂原文文句、乱码或其他语言
            if re.search(r'[^\u4e00-\u9fff\w\s.,!?;:]', trans_line) or orig_line in trans_line:
                issue = "译文夹杂原文文句、乱码或其他语言"
                new_text = retranslate_sentence(orig_line, style, api_key, model_url)
                if new_text and not re.search(r'[^\u4e00-\u9fff\w\s.,!?;:]', new_text) and orig_line not in new_text:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, True, new_text)
                    corrected_lines.append(f"[{sentence_index}][改] {new_text}")
                else:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, False)
                    corrected_lines.append(f"[{sentence_index}][改] {trans_line}")
                continue

            # 检查 4: 译文是否流畅（简单检查重复词或不完整句子）
            words = re.split(r'[,.!?;:]', trans_line)
            if any(word.strip() == "" for word in words) or len(set(words)) < len(words) / 2:
                issue = "译文不流畅，可能有语病或不完整"
                new_text = retranslate_sentence(orig_line, style, api_key, model_url)
                if new_text and len(set(re.split(r'[,.!?;:]', new_text))) >= len(re.split(r'[,.!?;:]', new_text)) / 2:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, True, new_text)
                    corrected_lines.append(f"[{sentence_index}][改] {new_text}")
                else:
                    log_translation_check(filename, sentence_index, orig_line, trans_line, issue, False)
                    corrected_lines.append(f"[{sentence_index}][改] {trans_line}")
                continue

            # 检查 5: 无问题，返回原始译文
            log_translation_check(filename, sentence_index, orig_line, trans_line)
            corrected_lines.append(trans_line)

    return "\n".join(corrected_lines) if corrected_lines else None

def count_tokens(text):
    encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(text))

def save_checkpoint(filename, start_sentence_index, start_chunk_index, translated_text, corrected_text):
    TEMP_CHECKPOINT_FILE = os.path.join(TEMP_DIR, f"{filename}_checkpoint.json")
    checkpoint_data = {
        "sentence_index": start_sentence_index,
        "chunk_index": start_chunk_index,
        "translated_text": translated_text,
        "checked_text": corrected_text
    }
    try:
        with open(TEMP_CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
        
    except Exception as e:
        logging.error(f"保存检查点失败: {e}")

def translate_document(file_path: str, style: str, temperature: float, model_url: str, api_key: str, batch_size_check: int):
    global is_thread_running, is_translating, global_file_path, global_style, global_temperature, global_batch_size, global_batch_size_check
    try:
        text = read_document(file_path)
        if text is None:
            logging.error("读取到的文本是 None")
            is_thread_running = False
            return
        
        filename = os.path.splitext(os.path.basename(file_path))[0]  # 先定义 filename
        log_handler = setup_specific_logging(filename)  # 使用 filename 初始化日志
        token_count = count_tokens(text)
        logging.info(f"文档总 token 数: {token_count}")
        
        translated_file_path = os.path.join(OUTPUT_DIR, f"{filename}_translated.txt")
        TEMP_INPUT_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_input.txt")
        TEMP_NUMBERED_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_numbered.txt")
        TEMP_TRANSLATED_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_translated.txt")
        TEMP_CHECKED_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_checked.txt")
        TEMP_CHECKPOINT_FILE = os.path.join(TEMP_DIR, f"{filename}_checkpoint.json")
        TEMP_TRANSLATING_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_translating.txt")

        translated_text = ""
        corrected_text = ""
        start_sentence_index = 0
        start_chunk_index = 0

        # 加载已有翻译结果并初始化 translated_dict
        translated_dict = {}
        if os.path.exists(TEMP_TRANSLATED_FILE):
            try:
                with open(TEMP_TRANSLATED_FILE, "r", encoding="utf-8") as f:
                    translated_text = f.read()
                    for line in translated_text.splitlines():
                        match = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                        if match:
                            idx = int(match.group(1)) - 1
                            translated_dict[idx] = line
                logging.info("检测到已存在的翻译临时文件，加载已有翻译结果。")
            except Exception as e:
                logging.warning(f"读取翻译临时文件失败: {e}, 从头开始翻译")
                translated_text = ""
                translated_dict = {}

        if os.path.exists(TEMP_CHECKED_FILE):
            try:
                with open(TEMP_CHECKED_FILE, "r", encoding="utf-8") as f:
                    corrected_text = f.read()
                logging.info("检测到已存在的校对临时文件，从上次校对位置继续。")
            except Exception as e:
                logging.warning(f"读取校对临时文件失败: {e}, 从头开始校对")
                corrected_text = ""
        if os.path.exists(TEMP_CHECKPOINT_FILE):
            try:
                with open(TEMP_CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    start_sentence_index = checkpoint_data.get("sentence_index", 0)
                    start_chunk_index = checkpoint_data.get("chunk_index", 0)
                    translated_text = checkpoint_data.get("translated_text", translated_text)
                    corrected_text = checkpoint_data.get("checked_text", corrected_text)
                    logging.info(f"检测到检查点文件，从句子 {start_sentence_index} 和块 {start_chunk_index} 继续。")
            except Exception as e:
                logging.warning(f"读取检查点文件失败: {e}, 从头开始翻译和校对")
                start_sentence_index = 0
                start_chunk_index = 0

        # 保存原始输入文件
        if not os.path.exists(TEMP_INPUT_FILE):
            with open(TEMP_INPUT_FILE, "w", encoding="utf-8") as f:
                f.write(text)
            logging.info(f"已保存原始输入文件到 {TEMP_INPUT_FILE}")

        # 第一步：生成带标号的原文文件
        sentences = text.splitlines()
        num_sentences = len(sentences)
        numbered_sentences = [f"[{i+1}] {sentence}" for i, sentence in enumerate(sentences)]
        numbered_text = "\n".join(numbered_sentences)
        if not os.path.exists(TEMP_NUMBERED_FILE) or os.path.getsize(TEMP_NUMBERED_FILE) == 0:
            try:
                with open(TEMP_NUMBERED_FILE, "w", encoding="utf-8") as f:
                    f.write(numbered_text)
                logging.info(f"已成功生成带标号的原文文件：{TEMP_NUMBERED_FILE}")
            except Exception as e:
                logging.error(f"生成带标号的原文文件失败：{e}")
                raise
        else:
            logging.info(f"带标号的原文文件已存在：{TEMP_NUMBERED_FILE}")

        # 验证文件内容
        with open(TEMP_NUMBERED_FILE, "r", encoding="utf-8") as f:
            numbered_content = f.read()
            
        # 更新全局文件路径为带标号的原文文件
        global_file_path = TEMP_NUMBERED_FILE
        global_batch_size_check = batch_size_check  # 设置用户指定的校对批次大小
        logging.info(f"翻译将使用带标号的原文文件：{global_file_path}, 校对批次大小：{batch_size_check}")

        open(IS_UPLOADING_FLAG, 'a').close()
        logging.info("创建上传标志文件")

        # 翻译阶段：直接从 TEMP_NUMBERED_FILE 读取
        with open(TEMP_NUMBERED_FILE, "r", encoding="utf-8") as f:
            numbered_sentences = f.read().splitlines()

        if start_sentence_index < len(numbered_sentences):
            for i in tqdm(range(start_sentence_index, len(numbered_sentences), global_batch_size), desc="翻译", initial=start_sentence_index):
                if not is_translating:
                    logging.info("翻译任务已暂停")
                    save_checkpoint(filename, i, start_chunk_index, translated_text, corrected_text)
                    break
                batch = numbered_sentences[i:i + global_batch_size]
                batch_text = "\n".join(batch)
                logging.info(f"当前处理句子批次：{i // global_batch_size + 1}/{len(numbered_sentences) // global_batch_size + 1}, 句子索引范围：{i}-{i + len(batch)-1}")
                save_checkpoint(filename, i, start_chunk_index, translated_text, corrected_text)

                # 翻译带标号的原文
                prompt = f"""请将以下文本翻译成流畅的简体字中文，并严格禁止拆分任何段落。完整翻译，禁止节译。只翻译，不解释。若遇古典诗句、颂文，按照严格工整的古诗格式翻译。禁止编造原文没有的内容（即使你认为原文的结尾还没完整）。正确使用标点符号。每句译文前保留原文句子编号（如 [1]），只返回翻译结果，不要包含原文、解释、说明或额外前缀后缀，确保输出只有带编号的翻译后的中文文本。翻译风格：{style}。
                待翻译的文本：
                {batch_text}"""
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature}
                }

                translated_lines = None
                for attempt in range(MAX_BATCH_RETRIES):
                    response = call_gemini_api(data, headers, model_url, api_key)
                    if response and "candidates" in response.json() and response.json()["candidates"]:
                        translated_batch_text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        translated_lines = translated_batch_text.splitlines()
                        logging.info(f"批量翻译完成，批次 {i // global_batch_size + 1} 返回 {len(translated_lines)} 句")
                        break
                    else:
                        logging.warning(f"翻译句子批次 {i // global_batch_size} 第 {attempt + 1} 次失败，重试中...")
                        if attempt == MAX_BATCH_RETRIES - 1:
                            logging.error(f"翻译句子批次 {i // global_batch_size} 达到最大重试次数 {MAX_BATCH_RETRIES}，暂停任务")
                            is_translating = False
                            save_checkpoint(filename, i, start_chunk_index, translated_text, corrected_text)
                            return
                        time.sleep(5)

                # 更新 translated_dict 仅覆盖新翻译结果
                if translated_lines:
                    for line in translated_lines:
                        match = re.match(r'\[(\d+)\]\s*(.*)', line.strip())
                        if match:
                            idx = int(match.group(1)) - 1  # 转换为 0-based 索引
                            translated_dict[idx] = line

                missing_indices = []
                for j, sentence in enumerate(batch):
                    sentence_index = i + j
                    orig_sentence = sentences[sentence_index]  # 获取原始句子用于比较
                    current_trans = translated_dict.get(sentence_index, None)  # 获取当前译文（可能为空或错误）

                    # 检查是否需要补译或覆盖
                    if (sentence_index not in translated_dict) or (current_trans and re.sub(r'\[\d+\]\s*', '', current_trans).strip() == orig_sentence.strip()):
                        retry_count = 0
                        max_retries = 5
                        while retry_count < max_retries:
                            prompt_single = f"""请将以下文本翻译成流畅的中文，并严格禁止拆分任何段落。完整翻译，禁止节译。只翻译，不解释。若遇藏文古典诗句、颂文，按照严格工整的汉语古诗格式翻译。禁止编造原文没有的内容。正确使用标点符号。只返回翻译结果。翻译风格：{style}。
                            {orig_sentence}"""
                            response_single = call_gemini_api({"contents": [{"parts": [{"text": prompt_single}]}]}, headers, model_url, api_key)
                            if response_single:
                                translated_single = response_single.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                                if translated_single.strip() != orig_sentence.strip():  # 补译成功
                                    translated_dict[sentence_index] = f"[{sentence_index + 1}] {translated_single}"
                                    if sentence_index in missing_indices:
                                        missing_indices.remove(sentence_index)
                                    logging.info(f"句子 {sentence_index} 补译或覆盖成功（尝试 {retry_count + 1} 次）")
                                    break
                            retry_count += 1
                            logging.warning(f"句子 {sentence_index} 第 {retry_count} 次补译失败或仍为原文，重试中...")
                            time.sleep(random.uniform(5, 10))

                        if retry_count == max_retries:  # 5 次补译失败
                            translated_dict[sentence_index] = f"[{sentence_index + 1}] [未翻译]"
                            if sentence_index not in missing_indices:
                                missing_indices.append(sentence_index)
                            logging.warning(f"句子 {sentence_index} 补译 5 次失败，标记为未翻译")
                    else:
                        # 检查长度差异并多次重译
                        orig_length = len(sentences[sentence_index])  # 原文字符数
                        trans_line = translated_dict[sentence_index]
                        trans_length = len(re.sub(r'\[\d+\]\s*', '', trans_line))  # 译文字符数（去标号）
                        length_ratio = max(orig_length, trans_length) / min(orig_length, trans_length) if min(orig_length, trans_length) > 0 else float('inf')
                        retry_count = 0
                        max_retries = 5  # 最大重试次数
                        while length_ratio > 5 and retry_count < max_retries:
                            prompt_single = f"""请将以下文本翻译成流畅的中文，并严格禁止拆分任何段落。完整翻译，禁止节译。只翻译，不解释。若遇藏文古典诗句、颂文，按照严格工整的汉语古诗格式翻译。禁止编造原文没有的内容。正确使用标点符号。只返回翻译结果。翻译风格：{style}。
                            {sentences[sentence_index]}"""
                            response_single = call_gemini_api({"contents": [{"parts": [{"text": prompt_single}]}]}, headers, model_url, api_key)
                            if response_single:
                                translated_single = response_single.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                                new_trans_length = len(translated_single)
                                new_length_ratio = max(orig_length, new_trans_length) / min(orig_length, new_trans_length) if min(orig_length, new_trans_length) > 0 else float('inf')
                                if new_length_ratio <= 5:
                                    translated_dict[sentence_index] = f"[{sentence_index + 1}] {translated_single}"
                                    logging.info(f"句子 {sentence_index} 长度差异过大（原文 {orig_length} 字，译文 {trans_length} 字），第 {retry_count + 1} 次重译成功，新长度 {new_trans_length} 字")
                                    break
                                else:
                                    trans_length = new_trans_length
                                    retry_count += 1
                                    logging.warning(f"句子 {sentence_index} 长度差异过大（原文 {orig_length} 字，译文 {trans_length} 字），第 {retry_count} 次重译仍不合理，重新尝试")
                                    if retry_count == max_retries:
                                        translated_dict[sentence_index] = f"[{sentence_index + 1}] {translated_single}"
                                        logging.warning(f"句子 {sentence_index} 重试 {max_retries} 次后仍长度异常，保留最后译文")
                            else:
                                logging.warning(f"句子 {sentence_index} 长度差异过大，第 {retry_count + 1} 次重译失败")
                                break
                            time.sleep(random.uniform(5, 10))

                # 保存所有译文（使用 translated_dict 保持已有翻译）
                batch_translated = [translated_dict.get(idx, f"[{idx + 1}] [未翻译]") for idx in range(0, len(sentences))]
                translated_text = "\n".join(batch_translated) + "\n"
                with open(TEMP_TRANSLATED_FILE, "w", encoding="utf-8") as f:
                    f.write(translated_text)
                if os.path.exists(TEMP_TRANSLATING_FILE):
                    os.remove(TEMP_TRANSLATING_FILE)
                save_checkpoint(filename, i + global_batch_size, start_chunk_index, translated_text, corrected_text)
                start_sentence_index = i + global_batch_size
                time.sleep(random.uniform(5, 10))

        # 新校对阶段：按标号逐段校对
        open(IS_CHECKING_FLAG, 'a').close()
        logging.info("创建校对标志文件")

        # 从文件读取原文和译文
        with open(TEMP_NUMBERED_FILE, "r", encoding="utf-8") as f:
            numbered_lines = f.read().splitlines()
        with open(TEMP_TRANSLATED_FILE, "r", encoding="utf-8") as f:
            translated_lines = f.read().splitlines()

        # 构建标号字典
        numbered_dict = {int(re.match(r'\[(\d+)\]', line).group(1)) - 1: line for line in numbered_lines if re.match(r'\[(\d+)\]', line)}
        translated_dict = {}
        for line in translated_lines:
            match = re.match(r'\[(\d+)\]', line.strip())
            if match:
                idx = int(match.group(1)) - 1
                translated_dict[idx] = line

        corrected_lines = []
        for i in tqdm(range(start_chunk_index, len(numbered_lines), batch_size_check), desc="逐段校对", initial=start_chunk_index // batch_size_check):
            if not is_translating:
                logging.info("校对任务已暂停")
                save_checkpoint(filename, start_sentence_index, i, translated_text, corrected_text)
                break
            batch_orig = [numbered_dict.get(j, f"[{j + 1}] [原文缺失]") for j in range(i, min(i + batch_size_check, len(numbered_lines)))]
            batch_trans = [translated_dict.get(j, f"[{j + 1}] [未翻译]") for j in range(i, min(i + batch_size_check, len(numbered_lines)))]

            batch_pairs = "\n".join(f"原文: {orig}\n译文: {trans}" for orig, trans in zip(batch_orig, batch_trans))
            corrected_chunk = check_translation(text, batch_pairs, style, filename, api_key, model_url)
            if corrected_chunk:
                corrected_lines.extend(corrected_chunk.splitlines())
                logging.info(f"校对批次 {i // batch_size_check + 1} 完成，返回 {len(corrected_chunk.splitlines())} 句")
            else:
                corrected_lines.extend(batch_trans)  # 如果校对失败，保留原始译文
                logging.warning(f"校对批次 {i // batch_size_check + 1} 失败，保留原始译文")
            time.sleep(random.uniform(5, 10))
            start_chunk_index = i + batch_size_check  # 更新校对进度

        corrected_text = "\n".join(corrected_lines) + "\n"
        with open(TEMP_CHECKED_FILE, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        logging.info(f"校对完成，结果保存到 {TEMP_CHECKED_FILE}")

        # 最终输出无标号文档
        cleaned_text = "\n".join(re.sub(r'\[\d+\](\[改\])?\s*', '', line.strip()) for line in corrected_text.splitlines())
        cleaned_text = clean_xml_string(cleaned_text).replace("\r\n", "\n").replace("\r", "\n")
        with codecs.open(translated_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logging.info(f"当前任务完成，无标号文件已保存到 {translated_file_path}")

        save_translation_params(filename)

    except Exception as e:
        logging.error(f"翻译过程中发生错误: {e}")
        save_checkpoint(filename, start_sentence_index, start_chunk_index, translated_text, corrected_text)
        is_thread_running = False
    finally:
        is_thread_running = False
        remove_specific_logging(log_handler)  # 移除日志处理器
        logging.info(f"特定日志已保存到 {current_log_file}")
        if is_translating:
            logging.info("当前任务已结束，未清理临时文件和标志文件")
            save_checkpoint(filename, start_sentence_index, start_chunk_index, translated_text, corrected_text)
        else:
            if os.path.exists(IS_UPLOADING_FLAG):
                os.remove(IS_UPLOADING_FLAG)
            if os.path.exists(IS_CHECKING_FLAG):
                os.remove(IS_CHECKING_FLAG)

def start_translation_thread(file_path, style, temperature, model_url, api_key, batch_size_check):
    global is_thread_running
    if not is_thread_running:
        is_thread_running = True
        executor.submit(translate_document, file_path, style, temperature, model_url, api_key, batch_size_check)
    else:
        logging.warning("翻译线程已经在运行，请稍后再试")

app = FastAPI()
static_folder_path = r"D:\gemini\geminitranslator\static"
if not os.path.exists(static_folder_path):
    raise RuntimeError(f"Directory '{static_folder_path}' does not exist")
app.mount("/static", StaticFiles(directory=static_folder_path), name="static")

@app.get("/", response_class=HTMLResponse)
def read_html():
    with open(os.path.join(static_folder_path, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/load_project_params")
async def load_project_params(project_name: str = Form(...)):
    params = load_translation_params(project_name)
    return params if params else {"file_path": "", "style": "", "temperature": ""}

@app.post("/load_project")
def load_project(file_path: str = Form(...), style: str = Form(...), temperature: float = Form(...)):
    global global_file_path, global_style, global_temperature
    logging.info(f"Load Project: Received - file_path: {file_path}, style: {style}, temperature: {temperature}")
    global_file_path = file_path
    global_style = style
    global_temperature = temperature
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_translation_params(filename)
    return {"message": f"已加载工程，文件路径：{file_path}"}

@app.post("/start_translation")
def start_translation(
    file_path: str = Form(...),
    style: str = Form(...),
    temperature: float = Form(...),
    api_key: str = Form(...),
    model_url: str = Form(...),
    batch_size: int = Form(...),
    batch_size_check: int = Form(...),
    http_proxy: str = Form(...),
    https_proxy: str = Form(...)
):
    global is_translating, global_file_path, global_style, global_temperature, global_batch_size, global_batch_size_check, PROXY
    logging.info(f"Start Translation: Received params - file_path: {file_path}, style: {style}, temperature: {temperature}, api_key: {api_key}, model_url: {model_url}, batch_size: {batch_size}, batch_size_check: {batch_size_check}, http_proxy: {http_proxy}, https_proxy: {https_proxy}")
    if not file_path:
        return {"message": "文件路径不能为空！"}
    if not api_key:
        return {"message": "API Key 不能为空！"}
    if not model_url:
        return {"message": "模型 URL 不能为空！"}
    if is_thread_running:
        return {"message": "翻译任务正在进行中，请稍等。"}
    
    is_translating = True
    global_file_path = file_path
    global_style = style
    global_temperature = temperature
    global_batch_size = batch_size  # 用户指定翻译批次大小
    global_batch_size_check = batch_size_check  # 用户指定校对批次大小
    PROXY = {"http": http_proxy, "https": https_proxy}
    logging.info(f"更新后的 PROXY: {PROXY}")

    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_translation_params(filename)
    start_translation_thread(file_path, style, temperature, model_url, api_key, batch_size_check)
    return {"message": "翻译任务已启动。"}

@app.post("/pause")
def pause_translation(file_path: str = Form(...), style: str = Form(...), temperature: float = Form(...)):
    global is_translating, global_file_path, global_style, global_temperature
    is_translating = False
    global_file_path = file_path
    global_style = style
    global_temperature = temperature
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_translation_params(filename)
    return {"message": "当前任务已暂停"}

@app.post("/resume")
def resume_translation(
    file_path: str = Form(...),
    style: str = Form(...),
    temperature: float = Form(...),
    api_key: str = Form(...),
    model_url: str = Form(...),
    batch_size: int = Form(...),
    batch_size_check: int = Form(...),
    http_proxy: str = Form(...),
    https_proxy: str = Form(...)
):
    global is_translating, global_file_path, global_style, global_temperature, global_batch_size, global_batch_size_check, PROXY
    logging.info(f"Resume: Received params - file_path: {file_path}, style: {style}, temperature: {temperature}, api_key: {api_key}, model_url: {model_url}, batch_size: {batch_size}, batch_size_check: {batch_size_check}, http_proxy: {http_proxy}, https_proxy: {https_proxy}")
    if is_thread_running:
        return {"message": "翻译任务正在进行中，请稍等。"}
    if not file_path:
        return {"message": "文件路径不能为空！"}
    if not api_key:
        return {"message": "API Key 不能为空！"}
    if not model_url:
        return {"message": "模型 URL 不能为空！"}
    
    global_file_path = file_path
    global_style = style
    global_temperature = temperature
    global_batch_size = batch_size
    global_batch_size_check = batch_size_check
    PROXY = {"http": http_proxy, "https": https_proxy}
    
    filename = os.path.splitext(os.path.basename(global_file_path))[0]
    TEMP_CHECKED_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_checked.txt")
    TEMP_TRANSLATED_FILE = os.path.join(TEMP_DIR, f"{filename}_temp_translated.txt")
    if os.path.exists(TEMP_CHECKED_FILE) or os.path.exists(TEMP_TRANSLATED_FILE):
        is_translating = True
        start_translation_thread(global_file_path, global_style, global_temperature, model_url, api_key, batch_size_check)
        return {"message": "任务已恢复，正在继续执行..."}
    return {"message": "没有检测到未完成的翻译或校对任务，请重新选择文件开始翻译。"}

@app.post("/stop")
def stop_translation(file_path: str = Form(...), style: str = Form(...), temperature: float = Form(...)):
    global is_translating, global_file_path, global_style, global_temperature
    logging.info(f"Stop Translation: Received file_path from form: {file_path}")
    is_translating = False
    global_file_path = file_path
    global_style = style
    global_temperature = temperature
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_translation_params(filename)
    for f in [
        os.path.join(TEMP_DIR, f"{filename}_temp_input.txt"),
        os.path.join(TEMP_DIR, f"{filename}_temp_translated.txt"),
        os.path.join(TEMP_DIR, f"{filename}_temp_translating.txt"),
        os.path.join(TEMP_DIR, f"{filename}_temp_checked.txt"),
        os.path.join(TEMP_DIR, f"{filename}_checkpoint.json"),
        os.path.join(TEMP_DIR, f"{filename}_temp_numbered.txt"),
        os.path.join(TEMP_DIR, f"{filename}_params.json"),
        IS_UPLOADING_FLAG,
        IS_CHECKING_FLAG
    ]:
        if os.path.exists(f):
            os.remove(f)
            logging.info(f"已删除文件: {f}")
    return {"message": "翻译任务已停止，所有进度已被清除。"}

@app.get("/unfinished_projects")
def list_unfinished_projects():
    projects = set()
    for file in os.listdir(TEMP_DIR):
        if any(file.endswith(suffix) for suffix in ["_checkpoint.json", "_temp_translated.txt", "_temp_checked.txt"]):
            proj = re.split(r'_temp_|_checkpoint', file)[0]
            projects.add(proj)
    return {"unfinished_projects": list(projects)} if projects else {"message": "未检测到未完成工程"}

@app.post("/select_project")
def select_project(project_name: str = Form(...), action: str = Form(...)):
    files = [
        os.path.join(TEMP_DIR, f"{project_name}_temp_input.txt"),
        os.path.join(TEMP_DIR, f"{project_name}_temp_translated.txt"),
        os.path.join(TEMP_DIR, f"{project_name}_temp_translating.txt"),
        os.path.join(TEMP_DIR, f"{project_name}_temp_checked.txt"),
        os.path.join(TEMP_DIR, f"{project_name}_temp_numbered.txt"),
        os.path.join(TEMP_DIR, f"{project_name}_checkpoint.json")
    ]
    if action == "continue":
        return {"message": f"将继续工程【{project_name}】的未完成任务"}
    elif action == "restart":
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        return {"message": f"已清理【{project_name}】的临时文件，将重新开始翻译任务"}
    return {"message": "无效的操作"}

def save_translation_params(filename):
    param_file = os.path.join(TEMP_DIR, f"{filename}_params.json")
    params = {
        "file_path": os.path.normpath(global_file_path) if global_file_path else "",
        "style": global_style,
        "temperature": global_temperature
    }
    os.makedirs(os.path.dirname(param_file), exist_ok=True)
    try:
        with open(param_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        logging.info(f"Save Params: 已保存翻译参数到: {param_file}, 内容: {params}")
    except Exception as e:
        logging.error(f"Save Params: 保存翻译参数失败: {e}")

def load_translation_params(filename):
    param_file = os.path.join(TEMP_DIR, f"{filename}_params.json")
    try:
        if os.path.exists(param_file):
            with open(param_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logging.warning(f"参数文件不存在: {param_file}")
            return None
    except Exception as e:
        logging.error(f"加载翻译参数失败: {e}")
        return None

def signal_handler(sig, frame):
    logging.info("程序收到退出信号，正在保存检查点...")
    if global_file_path:
        filename = os.path.splitext(os.path.basename(global_file_path))[0]
        save_checkpoint(filename, 0, 0, "", "")  # 简化处理，依赖循环内保存
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)