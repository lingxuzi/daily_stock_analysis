"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json
import time
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        tools_caption = ['macd', 'rsi', 'roc', 'stoch', 'willr']
        time_frame = state["time_frame"]
        messages = state.get("messages", [])
        report_content = []
        for tool, caption in zip(tools, tools_caption):
            tool_args = {"kline_data": copy.deepcopy(state["kline_data"])}
            tool_result = tool.invoke(tool_args)
            messages.append(
                ToolMessage(
                    tool_call_id=tool.name, content=json.dumps(tool_result)
                )
            )
            report_content.append(tool_result)

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
        #             "You must analyze technical indicators to support fast-paced trading execution.\n\n"
        #             f"⚠️ The indicator data provided is from a {time_frame} intervals, reflecting recent market behavior. "
        #             f"Here is the computed indicator data:\n{json.dumps(report_content, indent=2)}\n\n"
        #             "You must interpret this data quickly and accurately.\n\n"
        #         ),
        #         MessagesPlaceholder(variable_name="messages"),
        #     ]
        # )
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
