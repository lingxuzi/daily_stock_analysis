import os
import httpx
import logging
import time
import traceback
from analyzer import WebTradingAnalyzer
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def get_recommendations():
    recommendations = []
    url = 'https://stock.ai.hamuna.club/stocks/recommendations/latest'
    with httpx.Client() as client:
        response = client.post(url, json={})
        response.raise_for_status()
        response = response.json()
        if response['success'] == 'ok':
            recommendations = response['data']['recommendations']
            date = response['data']['date']
    return recommendations, date

def upload_recommendation_analysis(content, date_):
    logger.info('uploading recommendation analysis...')
    url = 'https://stock.ai.hamuna.club/recommendations/analysis'
    date = datetime.now()
    with httpx.Client() as client:
        for i in range(3):
            try:
                response = client.post(url, json={
                    'date': date.date().strftime('%Y-%m-%d') if not date_ else date_,
                    'time': date.time().strftime('%H:%M'),
                    'summary': content[:20] + '...',
                    'content': content
                })
                response.raise_for_status()
                response = response.json()
                if response['success'] == 'ok':
                    print(f"推荐分析上传成功")
                    break
                else:
                    print(f"推荐分析上传失败: {response['msg']}")
                    time.sleep(5)
            except Exception as e:
                print(f"推荐分析上传失败: {e}")
                time.sleep(5)

def load_config():
    load_dotenv(override=True)
    config = {
        'agent_llm_provider': os.getenv('AGENT_LLM_PROVIDER'),
        'agent_llm_model': os.getenv('AGENT_LLM_MODEL'),
        'agent_llm_temperature': float(os.getenv('AGENT_LLM_TEMPERATURE')),
        'agent_llm_base_url': os.getenv('AGENT_LLM_BASE_URL'),
        'agent_api_key': os.getenv('AGENT_API_KEY').split(','),
        'graph_llm_provider': os.getenv('GRAPH_LLM_PROVIDER'),
        'graph_llm_model': os.getenv('GRAPH_LLM_MODEL'),
        'graph_llm_temperature': float(os.getenv('GRAPH_LLM_TEMPERATURE')),
        'graph_llm_base_url': os.getenv('GRAPH_LLM_BASE_URL'),
        'graph_api_key': os.getenv('GRAPH_API_KEY').split(',')
    }
    print(config)
    return config

def upload(file):
    with httpx.Client() as client:
        with open(file, "rb") as f:
            files = {"file": f}
            for i in range(3):
                try:
                    response = client.post("https://img.remit.ee/api/upload", files=files, headers={
                        'Origin': 'https://img.remit.ee',
                        'referer': 'https://img.remit.ee'
                    }, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                    if data["success"]:
                        return 'https://img.remit.ee' + data["url"]
                except Exception as e:
                    print(f"图片上传失败: {e}")
                finally:
                    time.sleep(5)
            

def create_stock_dashboard(stock_code, stock_name, stock_analysis_results):
    decision_map = {
        'SHORT': '看空',
        'LONG': '看多',
    }
    pattern_image_url = upload('kline_chart.png')
    if not pattern_image_url:
        pattern_image_url = 'data:image/png;base64,' + stock_analysis_results["pattern_chart"]
    
    trend_image_url = upload('trend_graph.png')
    if not trend_image_url:
        trend_image_url = 'data:image/png;base64,' + stock_analysis_results["trend_chart"]
    dashboard_content = f'## 分析结果 -> {stock_code} {stock_name}\n\n'
    dashboard_content += f"### 📌 核心结论: {stock_analysis_results['final_decision']['decision']}\n\n"
    dashboard_content += f'**市场情绪分析**: {stock_analysis_results["sentiment_analysis"]}\n\n'
    dashboard_content += f'**技术指标分析**: {stock_analysis_results["technical_indicators"]}\n\n'
    dashboard_content += f'![image]({pattern_image_url})\n\n'
    dashboard_content += f'**K线形态分析**: {stock_analysis_results["pattern_analysis"]}\n\n'
    dashboard_content += f'![image]({trend_image_url})\n\n'
    dashboard_content += f'**趋势分析**: {stock_analysis_results["trend_analysis"]}\n\n'
    dashboard_content += f'**决策理由**: {stock_analysis_results["final_decision"]["justification"]}\n\n'
    dashboard_content += f'**推荐策略**: {stock_analysis_results["final_decision"]["recommendation_strategy"]}\n\n'

    print(dashboard_content)

    return dashboard_content

if __name__ == "__main__":
    config = load_config()
    for _ in range(3):
        try:
            recommendations, date_ = get_recommendations()
            if recommendations:
                logger.info(f"获取到 {len(recommendations)} 条推荐")
                break
        except Exception as e:
            logger.error(f"获取推荐失败: {e}")
            time.sleep(5)

    stock_codes = [(item['股票代码'], item['股票名称']) for item in recommendations][:10]

    
    full_content = f"# 🎯 {date_} 决策仪表盘\n\n"
    end_date = datetime.now().date().strftime('%Y-%m-%d')
    start_date = (datetime.now().date() - timedelta(days=100)).strftime('%Y-%m-%d')
    try:
        for i, (code, code_name) in enumerate(stock_codes):
            for _ in range(3):
                analyzer = WebTradingAnalyzer(config)
                results = analyzer.analyze_asset(
                    code,
                    start_date, end_date, "d")
                if 'full_results' in results:
                    full_content += create_stock_dashboard(code, code_name, results['full_results'])
                    full_content += "\n\n---\n\n"   
                    time.sleep(10)
                    break
                else:
                    logger.error(f"{code} 分析失败: {results.get('error', '未知错误')}")
                    time.sleep(5)

            print(f'{i+1}/{len(stock_codes)} {code_name} analysis completed.')
        print(full_content)
        upload_recommendation_analysis(full_content, date_)
    except Exception as e:
        logger.error(f"推荐分析失败: {traceback.format_exc()}")