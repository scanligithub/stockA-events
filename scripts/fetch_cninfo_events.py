import os
import re
import sys
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import fitz  # PyMuPDF
import polars as pl

# -------------------------------------------------------------------------
# 1. 基础配置与日志
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EASTMONEY_EVENT_ENGINE")

EASTMONEY_API_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ⭐️ 优化后的吸附关键字矩阵
EVENT_CONFIG = {
    "YJYG": {"name": "业绩预告", "f_node": "1", "keywords": ["预告", "快报", "业绩大幅", "利润预增", "扭亏"]},
    "ZJC":  {"name": "增减持", "f_node": "2", "keywords": ["减持", "增持", "大股东变动", "持股变动"]}, # 剔除"股份变动"噪音
    "HG":   {"name": "股份回购", "f_node": "3", "keywords": ["回购股份", "回购方案", "回购报告书", "实施回购"]},
    "GQJL": {"name": "股权激励", "f_node": "4", "keywords": ["激励计划", "股权激励", "限制性股票", "股票期权", "授予"]},
    "JJ":   {"name": "限售股解禁", "f_node": "5", "keywords": ["解除限售", "限售股上市", "解禁"]},
    "FHSZ": {"name": "分红送转", "f_node": "6", "keywords": ["分红", "派息", "利润分配", "分配方案", "分配预案", "送转", "权益分派"]}, # 补齐"利润分配"
    "DXZF": {"name": "定向增发", "f_node": "7", "keywords": ["定向增发", "定增", "非公开发行", "特定对象发行"]}
}

# -------------------------------------------------------------------------
# 2. 数值转化函数
# -------------------------------------------------------------------------
def parse_to_wan_yuan(num: float, unit_str: str) -> float:
    if "亿" in unit_str:
        return num * 10000.0
    if "万" in unit_str:
        return num
    return num / 10000.0

# -------------------------------------------------------------------------
# 3. 模糊宽容正则提取器
# -------------------------------------------------------------------------
def extract_yjyg(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    pct_match = re.search(r"(?:增|上升|增加).*?([0-9\.]+)%.*?(?:至|-|~|到).*?([0-9\.]+)%", text)
    if pct_match:
        v1, v2 = float(pct_match.group(1)), float(pct_match.group(2))
    else:
        pct_match_down = re.search(r"(?:降|下降|减少).*?([0-9\.]+)%.*?(?:至|-|~|到).*?([0-9\.]+)%", text)
        if pct_match_down:
            v1, v2 = -float(pct_match_down.group(1)), -float(pct_match_down.group(2))

    profit_match = re.search(r"(?:盈利|净利润).*?([0-9\.]+)(万元|亿元).*?(?:至|-|~|到).*?([0-9\.]+)(万元|亿元)", text)
    if profit_match:
        v3 = parse_to_wan_yuan(float(profit_match.group(1)), profit_match.group(2))
    return v1, v2, v3

def extract_zjc(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    direction = -1.0 if "减持" in text else 1.0
    ratio_match = re.search(r"(?:减持|增持).*?([0-9\.]+)%", text)
    if ratio_match:
        v1 = float(ratio_match.group(1)) * direction
        
    vol_match = re.search(r"(?:减持|增持).*?([0-9\.]+)(万股|股)", text)
    if vol_match:
        num = float(vol_match.group(1))
        v2 = num / 10000.0 if "万" not in vol_match.group(2) else num
    return v1, v2, 0.0

def extract_jj(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    ratio_match = re.search(r"占(?:总股本|比例).*?([0-9\.]+)%", text)
    if ratio_match:
        v1 = float(ratio_match.group(1))
    vol_match = re.search(r"(?:解除限售|上市流通).*?([0-9\.]+)(股|万股)", text)
    if vol_match:
        num = float(vol_match.group(1))
        v2 = num / 10000.0 if "万" not in vol_match.group(2) else num
    return v1, v2, 0.0

def extract_hg(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    amt_match = re.search(r"不低于.*?([0-9\.]+)(万元|亿元).*?不超过.*?([0-9\.]+)(万元|亿元)", text)
    if amt_match:
        v1 = parse_to_wan_yuan(float(amt_match.group(1)), amt_match.group(2))
        v2 = parse_to_wan_yuan(float(amt_match.group(3)), amt_match.group(4))
    
    price_match = re.search(r"价格.*?([0-9\.]+)元", text)
    if price_match:
        v3 = float(price_match.group(1))
    return v1, v2, v3

def extract_gqjl(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    vol_match = re.search(r"(?:授予|计划).*?([0-9\.]+)(万股|万份|股)", text)
    if vol_match:
        num = float(vol_match.group(1))
        v1 = num / 10000.0 if "万" not in vol_match.group(2) else num
    
    price_match = re.search(r"价格.*?([0-9\.]+)元", text)
    if price_match:
        v2 = float(price_match.group(1))
    return v1, v2, 0.0

def extract_fhsz(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    cash_match = re.search(r"10股派.*?([0-9\.]+)元", text)
    if cash_match:
        v1 = float(cash_match.group(1))
    
    bonus_match = re.search(r"送(?:红股)?([0-9\.]+)股", text)
    if bonus_match:
        v2 = float(bonus_match.group(1))
        
    trans_match = re.search(r"转增([0-9\.]+)股", text)
    if trans_match:
        v3 = float(trans_match.group(1))
    return v1, v2, v3

def extract_dxzf(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    amt_match = re.search(r"总额.*?不超过.*?([0-9\.]+)(万元|亿元)", text)
    if amt_match:
        v1 = parse_to_wan_yuan(float(amt_match.group(1)), amt_match.group(2))
    
    price_match = re.search(r"价格.*?([0-9\.]+)元", text)
    if price_match:
        v2 = float(price_match.group(1))
    return v1, v2, 0.0

EXTRACT_ROUTER = {
    "YJYG": extract_yjyg, "ZJC": extract_zjc, "JJ": extract_jj,
    "HG": extract_hg, "GQJL": extract_gqjl, "FHSZ": extract_fhsz, "DXZF": extract_dxzf
}

# -------------------------------------------------------------------------
# 4. 下载、文本转化与正则解析主函数 (1KB 长度卫检)
# -------------------------------------------------------------------------
def process_single_pdf(ann_item: dict, event_type: str) -> dict:
    codes = ann_item.get("codes", [])
    if codes and isinstance(codes, list):
        stock_code = codes[0].get("stock_code", "")
        stock_name = codes[0].get("short_name", "")
    else:
        stock_code, stock_name = "", ""
        
    title = ann_item.get("title", "")
    art_code = ann_item.get("art_code", "")
    
    raw_date = ann_item.get("notice_date") or ann_item.get("show_time") or ""
    notice_date = raw_date[:10] if len(raw_date) >= 10 else "2021-01-01"
    
    if not stock_code or not art_code:
        return {"code": stock_code, "name": stock_name, "date": notice_date, "title": title, "event_type": event_type, "val_1": 0.0, "val_2": 0.0, "val_3": 0.0}

    prefix = "H3" if stock_code.startswith("6") else ("H1" if stock_code.startswith("8") or stock_code.startswith("4") else "H2")
    
    pdf_url_1 = f"https://pdf.dfcfw.com/pdf/{prefix}_{art_code}_1.pdf"
    pdf_url_2 = f"https://pdf.dfcfw.com/pdf/{prefix}_{art_code}.pdf"
    
    val_1, val_2, val_3 = 0.0, 0.0, 0.0
    try:
        resp = requests.get(pdf_url_1, headers=HEADERS, timeout=15)
        # ⭐️ 核心改进：不仅看状态码 200，还要通过“文件长度 > 1000 字节”拦截 CDN 空响应
        if resp.status_code != 200 or len(resp.content) < 1000:
            resp = requests.get(pdf_url_2, headers=HEADERS, timeout=15)
            
        if resp.status_code == 200 and len(resp.content) >= 1000:
            doc = fitz.open(stream=resp.content, filetype="pdf")
            extracted_text = ""
            for i in range(min(3, len(doc))):
                extracted_text += doc[i].get_text()
            doc.close()

            flat_text = re.sub(r'[\s,，、人民币]', '', extracted_text)
            
            parser = EXTRACT_ROUTER.get(event_type)
            if parser:
                res = parser(flat_text)
                val_1, val_2, val_3 = res[0], res[1], (res[2] if len(res) > 2 else 0.0)
        else:
            # 优雅降级记录，不再触发 fitz 的 stream 空白解析报错
            logger.debug(f"CDN 未提供有效 PDF: {stock_code} {title}")
    except Exception as e:
        logger.warning(f"解析异常 {stock_code} {title}: {str(e)}")

    return {
        "code": stock_code, "name": stock_name, "date": notice_date,
        "title": title, "event_type": event_type,
        "val_1": val_1, "val_2": val_2, "val_3": val_3
    }

# -------------------------------------------------------------------------
# 5. 东财接口查询
# -------------------------------------------------------------------------
def fetch_announcements_by_year(year: int, event_type: str) -> list:
    results = []
    page = 1
    page_size = 100
    cfg = EVENT_CONFIG[event_type]
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"
    
    logger.info(f"开始查询 {year} 年【{cfg['name']}】公告列表...")
    while True:
        params = {
            "sr": "-1", "page_size": str(page_size), "page_index": str(page),
            "ann_type": "A", "client_source": "web", "f_node": cfg["f_node"],
            "begin_time": start_date, "end_time": end_date
        }
        try:
            resp = requests.get(EASTMONEY_API_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200: break
            ann_list = resp.json().get("data", {}).get("list", [])
            if not ann_list: break
            results.extend(ann_list)
            if len(ann_list) < page_size or page >= 30: break
            page += 1
        except Exception as e:
            break
    logger.info(f"扫网完毕！原始描述: {len(results)} 份。")
    return results

# -------------------------------------------------------------------------
# 6. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    
    target_year = args.year
    os.makedirs("output_events", exist_ok=True)
    all_final_records = []
    
    for et_code, et_meta in EVENT_CONFIG.items():
        raw_announcements = fetch_announcements_by_year(target_year, et_code)
        if not raw_announcements: continue
            
        filtered_announcements = []
        keywords = EVENT_CONFIG[et_code]["keywords"]
        for item in raw_announcements:
            title = item.get("title", "")
            if any(kw in title for kw in keywords):
                filtered_announcements.append(item)
                
        logger.info(f"【降维过滤】: 剩余 {len(filtered_announcements)} 篇核心 PDF")
        if not filtered_announcements: continue
            
        logger.info("启动并发提取引擎...")
        records_part = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_single_pdf, item, et_code): item for item in filtered_announcements}
            processed_cnt = 0
            for fut in as_completed(futures):
                res = fut.result()
                if res["val_1"] != 0.0 or res["val_2"] != 0.0 or res["val_3"] != 0.0:
                    records_part.append(res)
                processed_cnt += 1
                if processed_cnt % 50 == 0:
                    logger.info(f"进度: {processed_cnt} / {len(filtered_announcements)}")
                    
        all_final_records.extend(records_part)
        logger.info(f"  -> 【{et_meta['name']}】产出真 Alpha 记录: {len(records_part)} 条\n")
        
    if all_final_records:
        df = pl.DataFrame(all_final_records)
        df = df.with_columns([
            pl.col("code").cast(pl.Utf8), pl.col("name").cast(pl.Utf8), pl.col("date").cast(pl.Utf8),
            pl.col("title").cast(pl.Utf8), pl.col("event_type").cast(pl.Utf8),
            pl.col("val_1").cast(pl.Float64), pl.col("val_2").cast(pl.Float64), pl.col("val_3").cast(pl.Float64)
        ])
        output_file = f"output_events/event_{target_year}.parquet"
        df.write_parquet(output_file, compression="zstd")
        logger.info(f"【🎉 成功】已生成极纯净事件因子库: {output_file}，共计 {len(df)} 行特征！")
    else:
        logger.warning(f"该年份 {target_year} 未提取到有效公告数据。")

if __name__ == "__main__":
    main()
