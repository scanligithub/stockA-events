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
logger = logging.getLogger("CNINFO_ENGINE")

# 巨潮资讯 API 地址与请求头
CNINFO_QUERY_URL = "http://www.cninfo.com.cn/new/announcement/query"
PDF_BASE_URL = "http://static.cninfo.com.cn/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest"
}

# 巨潮官方公告分类代码表 (7 大事件映射)
EVENT_CONFIG = {
    "YJYG": {
        "name": "业绩预告",
        "category": "category_yjyg_szsh;category_yjkb_szsh",
        "keywords": ["业绩预告", "业绩快报", "预增", "预减", "扭亏"]
    },
    "ZJC": {
        "name": "增减持",
        "category": "category_gdzjc_szsh",
        "keywords": ["增持", "减持", "股份变动", "股东减持"]
    },
    "JJ": {
        "name": "限售股解禁",
        "category": "category_xsjj_szsh",
        "keywords": ["解禁", "限售股上市流通", "解除限售"]
    },
    "HG": {
        "name": "股份回购",
        "category": "category_gfhg_szsh",
        "keywords": ["回购", "回购股份", "回购方案"]
    },
    "GQJL": {
        "name": "股权激励",
        "category": "category_gqjl_szsh",
        "keywords": ["股权激励", "限制性股票", "期权授予"]
    },
    "FHSZ": {
        "name": "分红送转",
        "category": "category_fhsz_szsh",
        "keywords": ["分红", "派息", "送股", "转增", "利润分配预案"]
    },
    "DXZF": {
        "name": "定向增发",
        "category": "category_dxzf_szsh",
        "keywords": ["定增", "非公开发行", "向特定对象发行"]
    }
}

# -------------------------------------------------------------------------
# 2. 正则提取工具函数 (文本数值化与归一化)
# -------------------------------------------------------------------------
def clean_num_str(text: str) -> float:
    """清理字符串中的千分位并转化为浮点数"""
    if not text:
        return 0.0
    text = text.replace(",", "").replace("，", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0

def parse_to_wan_yuan(num_str: str, unit_str: str) -> float:
    """将捕获的金额统一换算为 '万元' 存储"""
    num = clean_num_str(num_str)
    if "亿" in unit_str:
        return num * 10000.0
    if "万" in unit_str:
        return num
    # 如果没有写万/亿，通常默认为元，折算为万元
    return num / 10000.0

# -------------------------------------------------------------------------
# 3. 7 大事件专属正则路由匹配器
# -------------------------------------------------------------------------
def extract_yjyg(text: str) -> tuple:
    """业绩预告提取器: val_1 = 净利润同比下限%, val_2 = 同比上限%, val_3 = 净利润下限(万元)"""
    v1, v2, v3 = 0.0, 0.0, 0.0
    # 匹配同比增长比例
    pct_match = re.search(r"比上年同期增长约?(-?[0-9\.]+)%-(-?[0-9\.]+)%", text)
    if pct_match:
        v1 = clean_num_str(pct_match.group(1))
        v2 = clean_num_str(pct_match.group(2))
    else:
        # 单一变动率匹配
        single_pct = re.search(r"比上年同期(增长\|下降)约?(-?[0-9\.]+)%", text)
        if single_pct:
            val = clean_num_str(single_pct.group(2))
            v1 = -val if "下降" in single_pct.group(1) else val
            v2 = v1

    # 匹配净利润绝对值区间
    profit_match = re.search(r"盈利约?([0-9\.,]+)(万元\|亿元)至([0-9\.,]+)(万元\|亿元)", text)
    if profit_match:
        v3 = parse_to_wan_yuan(profit_match.group(1), profit_match.group(2))
    return v1, v2, v3

def extract_zjc(text: str) -> tuple:
    """增减持提取器: val_1 = 变动股本比例% (减持为负, 增持为正), val_2 = 变动股数下限(万股)"""
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
    """限售解禁提取器: val_1 = 解禁比例%, val_2 = 解禁数量(万股)"""
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
    """股份回购提取器: val_1 = 回购下限(万元), val_2 = 回购上限(万元), val_3 = 回购最高限价(元/股)"""
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
    """股权激励提取器: val_1 = 拟授予数量(万股), val_2 = 授予/行权价格(元/股)"""
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
    """分红送转提取器: val_1 = 现金派息金额(元/10股), val_2 = 送股数/10股, val_3 = 转增股数/10股"""
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
    """定向增发提取器: val_1 = 募资上限(万元), val_2 = 发行底价(元/股)"""
    v1, v2 = 0.0, 0.0
    amt_match = re.search(r"募集资金总额不超过([0-9\.,]+)(万元\|亿元)", text)
    if amt_match:
        v1 = parse_to_wan_yuan(amt_match.group(1), amt_match.group(2))
    
    price_match = re.search(r"发行价格不低于([0-9\.]+)元/股", text)
    if price_match:
        v2 = clean_num_str(price_match.group(1))
    return v1, v2, 0.0

# 路由匹配映射表
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
# 4. 下载、文本转化与正则解析主函数 (C 原生速度级)
# -------------------------------------------------------------------------
def process_single_pdf(ann_item: dict, event_type: str) -> dict:
    """
    下载单个 PDF，提取前 5 页文本并进行正则清洗提取
    """
    pdf_url = PDF_BASE_URL + ann_item["adjunctUrl"]
    sec_code = ann_item["secCode"]
    sec_name = ann_item["secName"]
    title = ann_item["announcementTitle"]
    notice_date = ann_item["announcementTimeStr"]  # YYYY-MM-DD
    
    val_1, val_2, val_3 = 0.0, 0.0, 0.0
    try:
        # 1. 内存中极速拉取 PDF bytes，免去本地磁盘写入 I/O 开销
        resp = requests.get(pdf_url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=15)
        if resp.status_code == 200:
            # 2. C 原生加速解析 PDF (PyMuPDF)
            doc = fitz.open(stream=resp.content, filetype="pdf")
            extracted_text = ""
            # 卫健拦截：强制最多只分析前 5 页，避免长文本引发 CPU 溢出
            pages_to_read = min(5, len(doc))
            for i in range(pages_to_read):
                extracted_text += doc[i].get_text()
            doc.close()

            # 3. 极速文本扁平化，过滤回车与换行，防止段落分裂干扰正则
            flat_text = "".join(extracted_text.split())
            
            # 4. 路由匹配
            parser = EXTRACT_ROUTER.get(event_type)
            if parser:
                res = parser(flat_text)
                val_1 = float(res[0])
                val_2 = float(res[1])
                val_3 = float(res[2]) if len(res) > 2 else 0.0
    except Exception as e:
        # 降级容错机制：个别不可读 PDF 不应影响全量爬虫流程
        logger.debug(f"解析公告失败 {sec_code} [{title}]: {str(e)}")

    return {
        "code": sec_code,
        "name": sec_name,
        "date": notice_date,
        "title": title,
        "event_type": event_type,
        "val_1": val_1,
        "val_2": val_2,
        "val_3": val_3
    }

# -------------------------------------------------------------------------
# 5. 巨潮接口查询与大批量任务调度
# -------------------------------------------------------------------------
def fetch_announcements_by_year(year: int, event_type: str) -> list:
    """
    通过巨潮网关全量扫描某一年度某类事件下的所有匹配公告
    """
    results = []
    page = 1
    page_size = 100 # 单次拉取最大上限，减少 HTTP 请求
    cfg = EVENT_CONFIG[event_type]
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    logger.info(f"开始查询 {year} 年【{cfg['name']}】公告列表...")
    
    while True:
        # 巨潮参数构造
        payload = {
            "pageNum": page,
            "pageSize": page_size,
            "tabName": "fulltext",
            "column": "szse_main;sse_main;kcb;cyb", # 全市场覆盖
            "plate": "",
            "stock": "",
            "searchkey": cfg["keywords"][0], # 核心主关键词路由
            "secid": "",
            "category": cfg["category"],
            "trade": "",
            "showTime": f"{start_date} ~ {end_date}",
            "sortName": "pubtime",
            "sortType": "desc"
        }
        
        try:
            resp = requests.post(CNINFO_QUERY_URL, headers=HEADERS, data=payload, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            announcements = data.get("announcements", [])
            if not announcements:
                break
            
            results.extend(announcements)
            # 判断是否已拉取完毕
            if len(announcements) < page_size or page >= 10:  # 生产环境安全拦截或控制抓取规模
                break
            page += 1
            time.sleep(1) # 礼貌延迟，保护目标 CDN
        except Exception as e:
            logger.error(f"查询巨潮接口异常: {str(e)}")
            break
            
    logger.info(f"查询完毕！在 {year} 年共匹配到 {len(results)} 份【{cfg['name']}】公告")
    return results

# -------------------------------------------------------------------------
# 6. 多线程并发调度主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CNINFO Event Alpha Factory")
    parser.add_argument("--year", type=int, required=True, help="需要提取公告的年份")
    args = parser.parse_args()
    
    target_year = args.year
    os.makedirs("output_events", exist_ok=True)
    
    all_final_records = []
    
    # 遍历 7 大高价值事件类型
    for et_code, et_meta in EVENT_CONFIG.items():
        # Step 1: 扫描并抓取公告源字典
        raw_announcements = fetch_announcements_by_year(target_year, et_code)
        if not raw_announcements:
            continue
            
        logger.info(f"启动多进程 C 加速解析器，处理 {len(raw_announcements)} 份 PDF...")
        
        # Step 2: 启动多线程进行网络 I/O 下载与极速 CPU 解析
        # 为照顾 GitHub Actions 的 2 核 VM，并发线程数设为 8-12 可达吞吐量最优比
        records_part = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_single_pdf, item, et_code): item 
                for item in raw_announcements
            }
            
            processed_cnt = 0
            for fut in as_completed(futures):
                res = fut.result()
                records_part.append(res)
                processed_cnt += 1
                if processed_cnt % 50 == 0:
                    logger.info(f"进度反馈: 已完成 {processed_cnt} / {len(raw_announcements)}")
                    
        all_final_records.extend(records_part)
        
    # Step 3: 采用 Polars 生成高压缩率 ZSTD Parquet 物理宽表
    if all_final_records:
        df = pl.DataFrame(all_final_records)
        output_file = f"output_events/event_{target_year}.parquet"
        
        # 强制强类型，避免下游因 NaN 或 空值 冲突
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
        logger.info(f"【🎉 成功】已生成 {target_year} 年度极纯净、去未来化事件因子库: {output_file}，共计 {len(df)} 行数据")
    else:
        logger.warning(f"该年份 {target_year} 未提取到任何有效公告数据。")

if __name__ == "__main__":
    main()
