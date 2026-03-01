"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json
import httpx
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from fake_useragent import UserAgent
from datetime import datetime, timedelta


def _fetch(
    url: str, params: dict = {}, headers: dict = None, timeout: int = 60, origin=True) -> dict:
    """通用异步GET请求（每次请求使用随机User-Agent）"""
    # 每次请求使用新的随机User-Agent
    request_headers = {"User-Agent": UserAgent().random, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"}
    if headers:
        request_headers.update(headers)
    try:
        response = httpx.get(
            url,
            params=params,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "text": response.text,
            "json": response.json()
            if "application/json" in response.headers.get("content-type")
            else None,
        }
            
    except Exception as e:
        print(f"[API] 请求失败 {url}: {e}")
        return None

def get_news_from_stock(code, days=7):
    print(f"[API] 获取 {code} 相关新闻...")
    """获取股票相关新闻"""
    try:
        code_str = str(code).strip()
        if code_str.startswith("6"):
            secid = f"1.{code_str}"
        elif code_str.startswith(("0", "3")):
            secid = f"0.{code_str}"
        else:
            secid = f"1.{code_str}"

        url = "https://np-listapi.eastmoney.com/comm/web/getListInfo"
        params = {
            "cfh": "1",
            "client": "web",
            "mTypeAndCode": secid,
            "type": "1",
            "pageSize": "50",
        }

        response = _fetch(
            url,
            params=params,
            headers={
                "Referer": f"https://quote.eastmoney.com/sh{code}.html"
                if code.startswith("6")
                else f"https://quote.eastmoney.com/sz{code}.html",
                "Accept": "*/*",
            },
            timeout=10,
        )

        if response["status_code"] == 200:
            try:
                data = json.loads(response["text"])
                if isinstance(data, dict) and "data" in data and "list" in data["data"]:
                    items = data["data"]["list"]
                    news_list = []
                    for item in items:
                        if isinstance(item, dict):
                            news_item = {
                                "title": item.get("Art_Title", ""),
                                "url": item.get("Art_Url", ""),
                                "summary": "",
                                "source": "东方财富",
                                "time": item.get("Art_ShowTime", ""),
                                "type": "news",
                            }
                            if news_item["title"]:
                                news_list.append(news_item)

                    if days > 0:
                        filtered_news = []
                        today = datetime.now().date()
                        for news in news_list:
                            try:
                                if news.get("time"):
                                    news_time = datetime.strptime(
                                        news["time"], "%Y-%m-%d %H:%M:%S"
                                    )
                                    news_date = news_time.date()
                                    days_ago_date = today - timedelta(days=days)
                                    if news_date >= days_ago_date:
                                        filtered_news.append(news)
                                else:
                                    filtered_news.append(news)
                            except:
                                filtered_news.append(news)
                        news_list = filtered_news

                    return news_list
            except:
                pass

        return []
    except Exception as e:
        return []

def get_guba_posts(code, latest_count=10, hot_count=10):
    print(f"[API] 获取 {code} 股吧帖子（最新+热门）...")
    """获取股吧帖子（最新+热门）"""
    try:
        url = "https://gbapi.eastmoney.com/webarticlelist/api/Article/Articlelist"
        headers = {
            "Referer": f"https://guba.eastmoney.com/list,{code},99.html",
            "Accept": "application/json",
        }

        all_posts = []
        post_ids = set()

        params_latest = {
            "code": code,
            "sorttype": "1",
            "ps": str(latest_count),
            "from": "CommonBaPost",
            "deviceid": "quoteweb",
            "version": "200",
            "product": "Guba",
            "plat": "Web",
            "needzd": "true",
        }

        response_latest = _fetch(
            url, params=params_latest, headers=headers, timeout=10
        )
        if response_latest["status_code"] == 200:
            try:
                data = json.loads(response_latest["text"])
                if (
                    isinstance(data, dict)
                    and "re" in data
                    and isinstance(data["re"], list)
                ):
                    for article in data["re"]:
                        if isinstance(article, dict):
                            post_id = article.get("post_id")
                            if post_id and post_id not in post_ids:
                                post_ids.add(post_id)
                                post_item = {
                                    "post_id": post_id,
                                    "title": article.get("post_title", ""),
                                    "url": article.get("post_url", ""),
                                    "author": article.get("user_nickname", ""),
                                    "read_count": article.get("post_click_count", 0),
                                    "comment_count": article.get(
                                        "post_comment_count", 0
                                    ),
                                    "time": article.get("post_publish_time", ""),
                                    "type": "forum",
                                    "sort_type": "latest",
                                }
                                if post_item["title"]:
                                    all_posts.append(post_item)
            except json.JSONDecodeError:
                pass

        params_hot = {
            "code": code,
            "sorttype": "2",
            "ps": str(hot_count),
            "from": "CommonBaPost",
            "deviceid": "quoteweb",
            "version": "200",
            "product": "Guba",
            "plat": "Web",
            "needzd": "true",
        }

        response_hot = _fetch(url, params=params_hot, headers=headers, timeout=10)
        if response_hot["status_code"] == 200:
            try:
                data = json.loads(response_hot["text"])
                if (
                    isinstance(data, dict)
                    and "re" in data
                    and isinstance(data["re"], list)
                ):
                    for article in data["re"]:
                        if isinstance(article, dict):
                            post_id = article.get("post_id")
                            if post_id and post_id not in post_ids:
                                post_ids.add(post_id)
                                post_item = {
                                    "post_id": post_id,
                                    "title": article.get("post_title", ""),
                                    "url": article.get("post_url", ""),
                                    "author": article.get("user_nickname", ""),
                                    "read_count": article.get("post_click_count", 0),
                                    "comment_count": article.get(
                                        "post_comment_count", 0
                                    ),
                                    "time": article.get("post_publish_time", ""),
                                    "type": "forum",
                                    "sort_type": "hot",
                                }
                                if post_item["title"]:
                                    all_posts.append(post_item)
            except json.JSONDecodeError:
                pass

        return all_posts
    except Exception as e:
        return []

def create_sentiment_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def sentiment_agent_node(state):

        symbol = state["stock_name"]
        news_list = get_news_from_stock(symbol)
        social_posts = get_guba_posts(symbol)
        
        prompt = f"""
            You are a professional financial sentiment analyst assistant operating under time-sensitive conditions.
            You must analyze news reports and social media posts to support fast-paced trading execution.
            ⚠️ The data provided reflecting recent market behavior.
            Here is the news data:
            {news_list}
            Here is the social media data:
            {social_posts}
            You must interpret this data quickly and accurately.
            Please output the summary of the news and social media posts and give the sentiment analysis.
            **语言规范**: 要求以中文输出
        """
        response = llm.invoke(prompt)

        return {
            "messages": [],
            "sentiment_report": response.content
        }

    return sentiment_agent_node
