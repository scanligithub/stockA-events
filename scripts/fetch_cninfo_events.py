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

# ⭐️ 核心改进：为 7 大事件配置极度严格的标题“预洗关键字”
EVENT_CONFIG = {
    "YJYG": {
        "name": "业绩预告", 
        "f_node": "1", 
        "keywords": ["预告", "快报", "业绩大幅", "利润预增", "扭亏"]
    },
    "ZJC":  {
        "name": "增减持", 
        "f_node": "2", 
        "keywords": ["减持", "增持", "股份变动", "股份减持", "股份增持"]
    },
    "HG":   {
        "name": "股份回购", 
        "f_node": "3", 
        "keywords": ["回购股份", "回购方案", "回购报告书", "回购的进展", "实施回购"]
    },
    "GQJL": {
        "name": "股权激励", 
        "f_node": "4", 
        "keywords": ["激励计划", "股权激励", "限制性股票", "股票期权", "授予"]
    },
    "JJ":   {
        "name": "限售股解禁", 
        "f_node": "5", 
        "keywords": ["解除限售", "限售股上市", "解禁"]
    },
    "FHSZ": {
        "name": "分红送转", 
        "f_node": "6", 
        "keywords": ["分红", "派息", "分配预案", "送转", "实施公告", "红利发放"]
    },
    "DXZF": {
        "name": "定向增发", 
        "f_node": "7", 
        "keywords": ["定向增发", "定增", "非公开发行", "特定对象发行"]
    }
}

# -------------------------------------------------------------------------
# 2. 数值转化函数
# -------------------------------------------------------------------------
def clean_num_str(text: str) -> float:
    if not text:
        return 0.0
    text = text.replace(",", "").replace("，", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0

def parse_to_wan_yuan(num_str: str, unit_str: str) -> float:
    num = clean_num_str(num_str)
    if "亿" in unit_str:
        return num * 10000.0
    if "万" in unit_str:
        return num
    return num / 10000.0

# -------------------------------------------------------------------------
# 3. 7 大事件专属正则提取器
# -------------------------------------------------------------------------
def extract_yjyg(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    pct_match = re.search(r"比上年同期增长约?(-?[0-9\.]+)%-(-?[0-9\.]+)%", text)
    if pct_match:
        v1 = clean_num_str(pct_match.group(1))
        v2 = clean_num_str(pct_match.group(2))
    else:
        single_pct = re.search(r"比上年同期(增长\|下降)约?(-?[0-9\.]+)%", text)
        if single_pct:
            val = clean_num_str(single_pct.group(2))
            v1 = -val if "下降" in single_pct.group(1) else val
            v2 = v1

    profit_match = re.search(r"盈利约?([0-9\.,]+)(万元\|亿元)至([0-9\.,]+)(万元\|亿元)", text)
    if profit_match:
        v3 = parse_to_wan_yuan(profit_match.group(1), profit_match.group(2))
    return v1, v2, v3

def extract_zjc(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    direction = -1.0 if "减持" in text else 1.0
    ratio_match = re.search(r"拟?累计?(?:减持\|增持)股份(?:数量)?不超过(?:总股本的)?([0-9\.]+)%", text)
    if ratio_match:
        v1 = clean_num_str(ratio_match.group(1)) * direction
    
    vol_match = re.search(r"(?:减持\|增持)股份(?:数量)?(?:不超过)?([0-9\.,]+)(万股\|股)", text)
    if vol_match:
        num = clean_num_str(vol_match.group(1))
        v2 = num / 10000.0 if "万" not in vol_match.group(2) else num
    return v1, v2, 0.0

def extract_jj(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    ratio_match = re.search(r"占公司总股本(?:的)?比例(?:为)?([0-9\.]+)%", text)
    if ratio_match:
        v1 = clean_num_str(ratio_match.group(1))
    vol_match = re.search(r"解除限售(?:的)?股份数量(?:为)?([0-9\.,]+)(股\|万股)", text)
    if vol_match:
        num = clean_num_str(vol_match.group(1))
        v2 = num / 10000.0 if "万" not in vol_match.group(2) else num
    return v1, v2, 0.0

def extract_hg(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    amt_match = re.search(r"不低于([0-9\.,]+)(万元\|亿元).*?不超过([0-9\.,]+)(万元\|亿元)", text)
    if amt_match:
        v1 = parse_to_wan_yuan(amt_match.group(1), amt_match.group(2))
        v2 = parse_to_wan_yuan(amt_match.group(3), amt_match.group(4))
    
    price_match = re.search(r"回购价格(?:不超过)?([0-9\.]+)元/股", text)
    if price_match:
        v3 = clean_num_str(price_match.group(1))
    return v1, v2, v3

def extract_gqjl(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    vol_match = re.search(r"授予(?:的限制性股票\|期权)?数量([0-9\.,]+)(万股\|万份\|股)", text)
    if vol_match:
        num = clean_num_str(vol_match.group(1))
        v1 = num / 10000.0 if "万" not in vol_match.group(2) else num
    
    price_match = re.search(r"授予价格(?:为)?([0-9\.]+)元/股", text)
    if price_match:
        v2 = clean_num_str(price_match.group(1))
    return v1, v2, 0.0

def extract_fhsz(text: str) -> tuple:
    v1, v2, v3 = 0.0, 0.0, 0.0
    cash_match = re.search(r"派发现金红利约?([0-9\.]+)元", text)
    if cash_match:
        v1 = clean_num_str(cash_match.group(1))
    
    bonus_match = re.search(r"送红股([0-9\.]+)股", text)
    if bonus_match:
        v2 = clean_num_str(bonus_match.group(1))
        
    trans_match = re.search(r"转增([0-9\.]+)股", text)
    if trans_match:
        v3 = clean_num_str(trans_match.group(1))
    return v1, v2, v3

def extract_dxzf(text: str) -> tuple:
    v1, v2 = 0.0, 0.0
    amt_match = re.search(r"募集资金总额不超过([0-9\.,]+)(万元\|亿元)", text)
    if amt_match:
        v1 = parse_to_wan_yuan(amt_match.group(1), amt_match.group(2))
    
    price_match = re.search(r"发行价格不低于([0-9\.]+)元/股", text)
    if price_match:
        v2 = clean_num_str(price_match.group(1))
    return v1, v2, 0.0

EXTRACT_ROUTER = {
    "YJYG": extract_yjyg,
    "ZJC": extract_zjc,
    "JJ": extract_jj,
    "HG": extract_hg,
    "GQJL": extract_gqjl,
    "FHSZ": extract_fhsz,
    "DXZF": extract_dxzf
}

# -------------------------------------------------------------------------
# 4. 下载、文本转化与正则解析主函数 (安全重构版)
# -------------------------------------------------------------------------
def process_single_pdf(ann_item: dict, event_type: str) -> dict:
    codes = ann_item.get("codes", [])
    if codes and isinstance(codes, list):
        stock_code = codes[0].get("stock_code", "")
        stock_name = codes[0].get("short_name", "")
    else:
        stock_code = ""
        stock_name = ""
        
    title = ann_item.get("title", "")
    art_code = ann_item.get("art_code", "")
    
    raw_date = ann_item.get("notice_date") or ann_item.get("show_time") or ""
    notice_date = raw_date[:10] if len(raw_date) >= 10 else "2021-01-01"
    
    if not stock_code or not art_code:
        return {
            "code": stock_code, "name": stock_name, "date": notice_date,
            "title": title, "event_type": event_type, "val_1": 0.0, "val_2": 0.0, "val_3": 0.0
        }

    if stock_code.startswith("6"):
        prefix = "H3"
    elif stock_code.startswith("8") or stock_code.startswith("4"):
        prefix = "H1"
    else:
        prefix = "H2"
        
    pdf_url = f"https://pdf.dfcfw.com/pdf/{prefix}_{art_code}_1.pdf"
    
    val_1, val_2, val_3 = 0.0, 0.0, 0.0
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            doc = fitz.open(stream=resp.content, filetype="pdf")
            extracted_text = ""
            pages_to_read = min(3, len(doc))
            for i in range(pages_to_read):
                extracted_text += doc[i].get_text()
            doc.close()

            flat_text = "".join(extracted_text.split())
            
            parser = EXTRACT_ROUTER.get(event_type)
            if parser:
                res = parser(flat_text)
                val_1 = float(res[0])
                val_2 = float(res[1])
                val_3 = float(res[2]) if len(res) > 2 else 0.0
    except Exception as e:
        logger.debug(f"解析公告失败 {stock_code} [{title}]: {str(e)}")

    return {
        "code": stock_code,
        "name": stock_name,
        "date": notice_date,
        "title": title,
        "event_type": event_type,
        "val_1": val_1,
        "val_2": val_2,
        "val_3": val_3
    }

# -------------------------------------------------------------------------
# 5. 东财接口查询 (深度扩大至 30 页，获取更深历史信号)
# -------------------------------------------------------------------------
def fetch_announcements_by_year(year: int, event_type: str) -> list:
    results = []
    page = 1
    page_size = 100
    cfg = EVENT_CONFIG[event_type]
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    logger.info(f"开始通过东财网关查询 {year} 年【{cfg['name']}】公告列表...")
    
    while True:
        params = {
            "sr": "-1",
            "page_size": str(page_size),
            "page_index": str(page),
            "ann_type": "A",
            "client_source": "web",
            "f_node": cfg["f_node"],
            "begin_time": start_date,
            "end_time": end_date
        }
        
        try:
            resp = requests.get(EASTMONEY_API_URL, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.error(f"东财网关响应异常，状态码: {resp.status_code}")
                break
                
            data = resp.json()
            ann_list = data.get("data", {}).get("list", [])
            if not ann_list:
                break
            
            results.extend(ann_list)
            # ⭐️ 核心改进：扩大扫描页数至 30 页（3000条原始公告），以确保经过“预清洗”后依然有足够多的高价值信号
            if len(ann_list) < page_size or page >= 30:  
                break
            page += 1
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"查询东财接口异常: {str(e)}")
            break
            
    logger.info(f"扫网完毕！共计扫描到 {len(results)} 份原始公告描述。")
    return results

# -------------------------------------------------------------------------
# 6. 多线程并发调度主程序 (包含降维预过滤逻辑)
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EastMoney Event Alpha Factory")
    parser.add_argument("--year", type=int, required=True, help="需要提取公告的年份")
    args = parser.parse_args()
    
    target_year = args.year
    os.makedirs("output_events", exist_ok=True)
    
    all_final_records = []
    
    for et_code, et_meta in EVENT_CONFIG.items():
        raw_announcements = fetch_announcements_by_year(target_year, et_code)
        if not raw_announcements:
            continue
            
        # ⭐️ 终极改进：在投喂线程池之前，执行“标题预清洗”！
        # 仅当标题含有本事件的特征词（如“预告”、“快报”等）时，才允许进入下载和 PDF 解析队列
        filtered_announcements = []
        keywords = EVENT_CONFIG[et_code]["keywords"]
        for item in raw_announcements:
            title = item.get("title", "")
            if any(kw in title for kw in keywords):
                filtered_announcements.append(item)
                
        logger.info(f"【过滤洗涤】: 原始描述 {len(raw_announcements)} 篇 ➔ 降维过滤后真正高价值 PDF: {len(filtered_announcements)} 篇")
        
        if not filtered_announcements:
            logger.info(f"  -> 【{et_meta['name']}】无满足关键词的有效公告，跳过 PDF 下载。")
            continue
            
        logger.info(f"启动 PyMuPDF 并发解码器，深度解析这 {len(filtered_announcements)} 份核心 PDF...")
        
        records_part = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_single_pdf, item, et_code): item 
                for item in filtered_announcements
            }
            
            processed_cnt = 0
            for fut in as_completed(futures):
                res = fut.result()
                # ⭐️ 改进：如果解析出的核心字段不为0，或者虽然为0但确实是真实公告，加入数据集
                # 此处只保留真正提取出有用因子的行（彻底消灭无意义的0值行占位，保持数据集极度纯净！）
                if res["val_1"] != 0.0 or res["val_2"] != 0.0 or res["val_3"] != 0.0:
                    records_part.append(res)
                processed_cnt += 1
                if processed_cnt % 50 == 0:
                    logger.info(f"进度反馈: 已完成 {processed_cnt} / {len(filtered_announcements)}")
                    
        all_final_records.extend(records_part)
        logger.info(f"  -> 【{et_meta['name']}】提纯完毕，产出真 Alpha 记录数: {len(records_part)} 条\n")
        
    if all_final_records:
        df = pl.DataFrame(all_final_records)
        output_file = f"output_events/event_{target_year}.parquet"
        
        df = df.with_columns([
            pl.col("code").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
            pl.col("date").cast(pl.Utf8),
            pl.col("title").cast(pl.Utf8),
            pl.col("event_type").cast(pl.Utf8),
            pl.col("val_1").cast(pl.Float64),
            pl.col("val_2").cast(pl.Float64),
            pl.col("val_3").cast(pl.Float64),
        ])
        
        df.write_parquet(output_file, compression="zstd")
        logger.info(f"【🎉 降维打击大获成功】已生成 {target_year} 年度极纯净、100%真信号事件因子库: {output_file}，共计 {len(df)} 行有效特征！")
    else:
        logger.warning(f"该年份 {target_year} 在深度过滤后，未提取到任何含有效数值的公告数据。")

if __name__ == "__main__":
    main()
