import os
import re
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import polars as pl

# -------------------------------------------------------------------------
# 1. 基础配置
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DYNAMIC_RECALL_ENGINE")

EASTMONEY_API_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 7 大核心分类及对应的东财硬路由节点
EVENT_CONFIG = {
    "YJYG": {
        "name": "业绩预告", "f_node": "1",
        "keywords": r"业绩|净利润|预增|预减|扭亏|续亏|盈利|亏损|快报"
    },
    "ZJC": {
        "name": "增减持", "f_node": "2",
        "keywords": r"增持|减持|买入|卖出"
    },
    "HG": {
        "name": "股份回购", "f_node": "3",
        "keywords": r"回购"
    },
    "GQJL": {
        "name": "股权激励", "f_node": "4",
        "keywords": r"股权激励|限制性股票|股票期权|激励计划"
    },
    "JJ": {
        "name": "限售股解禁", "f_node": "5",
        "keywords": r"限售股|上市流通|解除限售"
    },
    "FHSZ": {
        "name": "分红送转", "f_node": "6",
        "keywords": r"利润分配|权益分派|分红|派息|派现|送股|转增"
    },
    "DXZF": {
        "name": "定向增发", "f_node": "7",
        "keywords": r"定向增发|定增|非公开发行|特定对象发行"
    }
}

# -------------------------------------------------------------------------
# 2. 动态分页拉取器 (带 f_node 路由)
# -------------------------------------------------------------------------
def fetch_metadata_by_page(page_index: int, f_node: str, start_date: str, end_date: str) -> list:
    params = {
        "sr": "-1",
        "page_size": "100",
        "page_index": str(page_index),
        "ann_type": "A",
        "client_source": "web",
        "f_node": f_node,
        "begin_time": start_date,
        "end_time": end_date
    }
    try:
        resp = requests.get(EASTMONEY_API_URL, params=params, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("list", [])
    except Exception:
        pass
    return []

def scan_channel_metadata(et_code: str, start_date: str, end_date: str) -> list:
    """动态探测单通道总页数并利用线程池 100% 扫网"""
    cfg = EVENT_CONFIG[et_code]
    f_node = cfg["f_node"]
    
    # 1. 探测第 1 页获取总记录数
    params = {
        "sr": "-1", "page_size": "100", "page_index": "1",
        "ann_type": "A", "client_source": "web", "f_node": f_node,
        "begin_time": start_date, "end_time": end_date
    }
    try:
        resp = requests.get(EASTMONEY_API_URL, params=params, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            logger.error(f"连接东财通道 {f_node} 失败！")
            return []
        raw_json = resp.json()
        total_hits = raw_json.get("data", {}).get("total", 0)
        first_page_list = raw_json.get("data", {}).get("list", [])
    except Exception as e:
        logger.error(f"探测通道 {f_node} 失败: {str(e)}")
        return []
        
    if total_hits == 0 or not first_page_list:
        return []
        
    # 动态自适应计算总页数 (单通道最多扫 50 页，防过度访问)
    total_pages = min(50, (total_hits + 99) // 100)
    logger.info(f"  -> 探测结果：通道【{cfg['name']}】共有历史公告 {total_hits} 条，准备并发扫描 {total_pages} 页...")
    
    all_metadata = []
    all_metadata.extend(first_page_list)
    
    if total_pages > 1:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_metadata_by_page, p, f_node, start_date, end_date) for p in range(2, total_pages + 1)]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    all_metadata.extend(res)
                    
    return all_metadata

# -------------------------------------------------------------------------
# 【主执行流：测试 2024年1月 核心数据】
# -------------------------------------------------------------------------
if __name__ == "__main__":
    start_t = "2024-01-01"
    end_t = "2024-01-31"
    
    os.makedirs("output_debug", exist_ok=True)
    all_recalled_records = []
    
    print("==================================================")
    print(f"🚀 启动 A 股信披大通道动态扫网战役 ({start_t} ~ {end_t})")
    print("==================================================")
    
    # 依次扫描 7 个大通道
    for et_code, et_meta in EVENT_CONFIG.items():
        # 1. 动态全量扫描单个通道
        raw_channel_data = scan_channel_metadata(et_code, start_t, end_t)
        if not raw_channel_data:
            print(f"  -> 【{et_meta['name']}】通道无数据。")
            continue
            
        # 2. 本地模糊标题过滤 (不进行 break 拦截)
        channel_recalled = []
        for item in raw_channel_data:
            title = item.get("title", "")
            codes = item.get("codes", [])
            stock_code = codes[0].get("stock_code", "") if codes else ""
            stock_name = codes[0].get("short_name", "") if codes else ""
            raw_date = item.get("notice_date") or item.get("show_time") or ""
            notice_date = raw_date[:10]
            
            if not stock_code:
                continue
                
            # 本地正则宽容筛选
            if re.search(et_meta["keywords"], title):
                channel_recalled.append({
                    "code": stock_code,
                    "name": stock_name,
                    "date": notice_date,
                    "title": title,
                    "event_type": et_code
                })
                
        print(f"  -> 【{et_meta['name']}】原始抓取: {len(raw_channel_data)} 条 ➔ 本地宽容过滤后真事件: {len(channel_recalled)} 条")
        all_recalled_records.extend(channel_recalled)
        
    print("\n--------------------------------------------------")
    print(f"🎉 扫网战役大获成功！2024年01月共计极限召回真事件: {len(all_recalled_records)} 条")
    print("--------------------------------------------------")
    
    # 3. 物理落盘
    if all_recalled_records:
        df = pl.DataFrame(all_recalled_records)
        output_file = "output_debug/title_recall_202401.parquet"
        df.write_parquet(output_file, compression="zstd")
        print(f"因子数据已成功导出至: {output_file}")
