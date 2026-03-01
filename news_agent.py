"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json
import time
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_news_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        
        prompt = f"""
            You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions.
            You must analyze technical indicators to support fast-paced trading execution.
            ⚠️ The indicator data provided is from a {time_frame} intervals, reflecting recent market behavior.
            Here is the computed indicator data:
            {report_content}
            You must interpret this data quickly and accurately.
            **语言规范**: 要求以中文输出
        """
        response = llm.invoke(prompt)

        return {
            "messages": messages,
            "indicator_report": response.content
        }

    return indicator_agent_node
