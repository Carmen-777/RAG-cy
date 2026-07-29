import os
import requests
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from src.page_markers import inject_pages_in_range, parse_page_range_end, parse_page_range_start

load_dotenv()
api_key = os.getenv("MINERU_API_KEY")
if not api_key:
    raise ValueError("未找到 MINERU_API_KEY，请在项目根目录 .env 文件中配置")

DEFAULT_PDF_OSS_BASE = "https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/"
DEFAULT_LOCAL_PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "stock_data" / "pdf_reports"


def _auth_headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def download_pdf_from_oss(file_name: str, local_dir: Path = DEFAULT_LOCAL_PDF_DIR) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / file_name
    if local_path.exists():
        print(f"本地 PDF 已存在，跳过下载: {local_path}")
        return local_path

    pdf_url = DEFAULT_PDF_OSS_BASE + quote(file_name)
    print(f"正在下载 PDF: {pdf_url}")
    response = requests.get(pdf_url, timeout=120)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    print(f"PDF 已保存到: {local_path}")
    return local_path


MAX_PAGES_PER_TASK = 200


def get_pdf_page_count(local_file_path: Path) -> int:
    return len(PdfReader(str(local_file_path)).pages)


def build_page_ranges(total_pages: int, max_pages: int = MAX_PAGES_PER_TASK) -> list[str]:
    ranges = []
    start = 1
    while start <= total_pages:
        end = min(start + max_pages - 1, total_pages)
        ranges.append(f"{start}-{end}")
        start = end + 1
    return ranges


def upload_local_file_and_get_batch_id(
    local_file_path: Path,
    data_id: str = "rag-step2",
    page_ranges: str | None = None,
) -> str:
    url = "https://mineru.net/api/v4/file-urls/batch"
    file_name = local_file_path.name
    file_meta = {"name": file_name, "data_id": data_id, "is_ocr": True}
    if page_ranges:
        file_meta["page_ranges"] = page_ranges
    data = {
        "files": [file_meta],
        "model_version": "vlm",
        "enable_formula": False,
    }

    res = requests.post(url, headers=_auth_headers(), json=data)
    body = res.json()
    print(res.status_code)
    print(body)
    if body.get("code") != 0 or "data" not in body:
        raise RuntimeError(f"MinerU 申请上传链接失败: {body.get('msg', body)}")

    batch_id = body["data"]["batch_id"]
    upload_urls = body["data"]["file_urls"]
    with open(local_file_path, "rb") as f:
        upload_res = requests.put(upload_urls[0], data=f)
    print(f"文件上传状态: {upload_res.status_code}")
    if upload_res.status_code != 200:
        raise RuntimeError(f"MinerU 文件上传失败: {upload_res.status_code} {upload_res.text}")
    print(f"文件上传成功, batch_id: {batch_id}")
    return batch_id


def get_batch_result(batch_id: str, extract_dir: str | None = None):
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    while True:
        res = requests.get(url, headers=_auth_headers())
        body = res.json()
        if body.get("code") != 0 or "data" not in body:
            raise RuntimeError(f"MinerU 查询批量任务失败: {body.get('msg', body)}")

        result = body["data"]
        print(result)
        extract_results = result.get("extract_result", [])
        if not extract_results:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue

        item = extract_results[0]
        state = item.get("state")
        err_msg = item.get("err_msg", "")
        if state in ["waiting-file", "pending", "running", "converting"]:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        if err_msg:
            raise RuntimeError(f"MinerU 解析失败: {err_msg}")
        if state == "done":
            full_zip_url = item.get("full_zip_url")
            if full_zip_url:
                local_filename = f"{batch_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                target_dir = extract_dir or batch_id
                unzip_file(local_filename, target_dir)
                return target_dir
            else:
                print("未找到 full_zip_url，无法下载。")
            return None
        raise RuntimeError(f"MinerU 返回未知状态: {state}")


def parse_pdf_by_name(
    file_name: str,
    local_pdf_dir: Path | None = None,
) -> Path:
    """推荐流程：优先使用本地 PDF，否则从 OSS 下载，再通过 MinerU 本地上传接口解析。"""
    local_dir = Path(local_pdf_dir) if local_pdf_dir else DEFAULT_LOCAL_PDF_DIR
    local_path = local_dir / file_name
    if not local_path.exists():
        local_path = download_pdf_from_oss(file_name, local_dir)

    total_pages = get_pdf_page_count(local_path)
    page_ranges_list = build_page_ranges(total_pages)
    print(f"PDF 共 {total_pages} 页，将分 {len(page_ranges_list)} 次解析: {page_ranges_list}")

    merged_dir = Path(f"{Path(file_name).stem}_merged")
    merged_dir.mkdir(exist_ok=True)
    merged_md_parts = []

    for idx, page_ranges in enumerate(page_ranges_list, start=1):
        print(f"\n=== 开始解析第 {idx}/{len(page_ranges_list)} 段: {page_ranges} ===")
        batch_id = upload_local_file_and_get_batch_id(
            local_path,
            data_id=f"rag-step2-part{idx}",
            page_ranges=page_ranges,
        )
        extract_dir = get_batch_result(batch_id, extract_dir=f"{batch_id}")
        md_path = Path(extract_dir) / "full.md"
        if not md_path.exists():
            raise RuntimeError(f"未找到解析结果: {md_path}")
        md_part = md_path.read_text(encoding="utf-8")
        start_page = parse_page_range_start(page_ranges)
        end_page = parse_page_range_end(page_ranges)
        merged_md_parts.append(inject_pages_in_range(md_part, start_page, end_page))

    merged_md_path = merged_dir / "full.md"
    merged_md_path.write_text("\n\n".join(merged_md_parts), encoding="utf-8")
    print(f"\n全部解析完成，合并 Markdown 已保存到: {merged_md_path}")
    return merged_md_path


def unzip_file(zip_path, extract_dir=None):
    if extract_dir is None:
        extract_dir = zip_path.rstrip(".zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")


if __name__ == "__main__":
    file_name = "【财报】中芯国际：中芯国际2024年年度报告.pdf"
    merged_md_path = parse_pdf_by_name(file_name)
    print("merged_md_path:", merged_md_path)
