import baostock as bs
from trading_graph import TradingGraph
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from flask import render_template
import urllib
import static_util

class WebTradingAnalyzer:
    def __init__(self, config):
        """Initialize the web trading analyzer."""
        self.trading_graph = TradingGraph(config=config)
        self.data_dir = Path("data")
        self.config = config

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

    def _format_code(self, code):
        if len(code) == 9:
            return code
        prefix = self._get_code_prefix(code)
        return f'{prefix}.{code}'
    
    def _get_code_prefix(self, stock_code):
        """
        根据A股股票代码前缀判断其所属的交易板块，并返回缩写形式。

        Args:
            stock_code (str): 6位数字的股票代码字符串。

        Returns:
            str: 描述股票所属板块的缩写信息，如果代码无效则返回错误提示。
            缩写说明：
            - SH_MAIN: 上海证券交易所主板
            - SZ_MAIN: 深圳证券交易所主板
            - SZ_CYB: 深圳证券交易所创业板
            - SH_KCB: 上海证券交易所科创板
            - BJ_EQ: 北京证券交易所股票
            - SH_B: 上海证券交易所B股
            - SZ_B: 深圳证券交易所B股
            - UNKNOWN: 未知板块或无效代码
        """
        if not isinstance(stock_code, str) or len(stock_code) != 6 or not stock_code.isdigit():
            return "UNKNOWN: Invalid Code"

        first_three_digits = stock_code[:3]
        first_two_digits = stock_code[:2]

        # A股主要板块判断
        if first_three_digits in ['600', '601', '603', '605']:
            return "SH"
        elif first_three_digits in ['000', '001', '002', '003']:
            return "SZ"
        elif first_three_digits == '300':
            return "SZ"
        elif first_three_digits == '688':
            return "SH"
        # 北交所代码判断：83, 87, 88开头，或从新三板平移的430开头
        elif first_two_digits in ['83', '87', '88'] or first_three_digits == '430':
            return "BJ"
        
        # B股判断
        elif first_three_digits == '900':
            return "SH"
        elif first_three_digits == '200':
            return "SZ"
            
        # 其他特殊代码或不常见代码，例如配股代码
        elif first_three_digits == '700':
            return "SH" # 沪市配股代码
        elif first_three_digits == '080':
            return "SZ" # 深市配股代码
        
        else:
            return "UNKNOWN"

    def fetch_finance_data(
        self, symbol: str, start_date: str, end_date: str, interval: str="1d"
    ) -> pd.DataFrame:
        symbol = self._format_code(symbol)
        bs.login()
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,code,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency=interval,
            adjustflag="3"
        )
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        df = pd.DataFrame(data_list, columns=rs.fields)
        df.rename(columns={
            "date": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }, inplace=True)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df[["Datetime", "Open", "High", "Low", "Close"]]
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)

        return df

    def get_available_assets(self) -> list:
        """Get list of available assets from the asset mapping dictionary."""
        return sorted(list(self.asset_mapping.keys()))

    def get_available_files(self, asset: str, timeframe: str) -> list:
        """Get available data files for a specific asset and timeframe."""
        asset_dir = self.data_dir / asset.lower()
        if not asset_dir.exists():
            return []

        pattern = f"{asset}_{timeframe}_*.csv"
        files = list(asset_dir.glob(pattern))
        return sorted(files)

    def run_analysis(
        self, df: pd.DataFrame, asset_name: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # Debug: Check DataFrame structure
            print(f"DataFrame columns: {df.columns}")
            print(f"DataFrame index: {type(df.index)}")
            print(f"DataFrame shape: {df.shape}")

            # Prepare data for analysis
            # if len(df) > 49:
            #     df_slice = df.tail(49).iloc[:-3]
            # else:
            #     df_slice = df.tail(45)

            df_slice = df.tail(60)

            # Ensure DataFrame has the expected structure
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df_slice.columns for col in required_columns):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available: {list(df_slice.columns)}",
                }

            # Reset index to avoid any MultiIndex issues
            df_slice = df_slice.reset_index(drop=True)

            # Debug: Check the slice before conversion
            print(f"Slice columns: {df_slice.columns}")
            print(f"Slice index: {type(df_slice.index)}")

            # Convert to dict for tool input - use explicit conversion to avoid tuple keys
            df_slice_dict = {}
            for col in required_columns:
                if col == "Datetime":
                    # Convert datetime objects to strings for JSON serialization
                    df_slice_dict[col] = (
                        df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                    )
                else:
                    df_slice_dict[col] = df_slice[col].tolist()

            # Debug: Check the resulting dictionary
            print(f"Dictionary keys: {list(df_slice_dict.keys())}")
            print(f"Dictionary key types: {[type(k) for k in df_slice_dict.keys()]}")

            # Format timeframe for display
            display_timeframe = timeframe
            if timeframe.endswith("h"):
                display_timeframe += "our"
            elif timeframe.endswith("m"):
                display_timeframe += "in"
            elif timeframe.endswith("d"):
                display_timeframe += "ay"
            elif timeframe == "1w":
                display_timeframe = "1 week"
            elif timeframe == "1mo":
                display_timeframe = "1 month"

            p_image = static_util.generate_kline_image(df_slice_dict)
            t_image = static_util.generate_trend_image(df_slice_dict)

            # Create initial state
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": display_timeframe,
                "stock_name": asset_name,
                "pattern_image": p_image["pattern_image"],
                "trend_image": t_image["trend_image"],
            }

            # Run the trading graph
            final_state = self.trading_graph.graph.invoke(initial_state)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": asset_name,
                "timeframe": display_timeframe,
                "data_length": len(df_slice),
            }

        except Exception as e:
            error_msg = str(e)
            
            # Get current provider from config
            provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            else:
                provider_name = "Qwen"

            # Check for specific API key authentication errors
            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "success": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key you provided is invalid or has expired. Please check your API key in the Settings section and try again.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "success": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection and try again.",
                }
            else:
                return {"success": False, "error": f"❌ Analysis Error: {error_msg}"}

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": results["error"]}

        final_state = results["final_state"]

        # Extract analysis results from state fields
        technical_indicators = final_state.get("indicator_report", "")
        pattern_analysis = final_state.get("pattern_report", "")
        trend_analysis = final_state.get("trend_report", "")
        final_decision_raw = final_state.get("final_trade_decision", "")

        # Extract chart data if available
        pattern_chart = final_state.get("pattern_image", "")
        trend_chart = final_state.get("trend_image", "")
        pattern_image_filename = final_state.get("pattern_image_filename", "")
        trend_image_filename = final_state.get("trend_image_filename", "")

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find("{")
                end = final_decision_raw.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": decision_data.get("decision", "N/A"),
                        "risk_reward_ratio": decision_data.get(
                            "risk_reward_ratio", "N/A"
                        ),
                        "forecast_horizon": decision_data.get(
                            "forecast_horizon", "N/A"
                        ),
                        "justification": decision_data.get("justification", "N/A"),
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": final_decision_raw}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": final_decision_raw}

        return {
            "success": True,
            "asset_name": results["asset_name"],
            "timeframe": results["timeframe"],
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision,
        }

    def get_timeframe_date_limits(self, timeframe: str) -> Dict[str, Any]:
        """Get valid date range limits for a given timeframe."""
        limits = {
            "1m": {"max_days": 7, "description": "1 minute data: max 7 days"},
            "2m": {"max_days": 60, "description": "2 minute data: max 60 days"},
            "5m": {"max_days": 60, "description": "5 minute data: max 60 days"},
            "15m": {"max_days": 60, "description": "15 minute data: max 60 days"},
            "30m": {"max_days": 60, "description": "30 minute data: max 60 days"},
            "60m": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "90m": {"max_days": 60, "description": "90 minute data: max 60 days"},
            "1h": {"max_days": 730, "description": "1 hour data: max 730 days"},
            "4h": {"max_days": 730, "description": "4 hour data: max 730 days"},
            "1d": {"max_days": 730, "description": "1 day data: max 730 days"},
            "5d": {"max_days": 60, "description": "5 day data: max 60 days"},
            "1w": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1wk": {"max_days": 730, "description": "1 week data: max 730 days"},
            "1mo": {"max_days": 730, "description": "1 month data: max 730 days"},
            "3mo": {"max_days": 730, "description": "3 month data: max 730 days"},
        }

        return limits.get(
            timeframe, {"max_days": 730, "description": "Default: max 730 days"}
        )

    def validate_date_range(
        self,
        start_date: str,
        end_date: str,
        timeframe: str,
        start_time: str = "00:00",
        end_time: str = "23:59",
    ) -> Dict[str, Any]:
        """Validate date and time range for the given timeframe."""
        try:
            # Create datetime objects with time
            start_datetime_str = f"{start_date} {start_time}"
            end_datetime_str = f"{end_date} {end_time}"

            start = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
            end = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")

            if start >= end:
                return {
                    "valid": False,
                    "error": "Start date/time must be before end date/time",
                }

            # Get timeframe limits
            limits = self.get_timeframe_date_limits(timeframe)
            max_days = limits["max_days"]

            # Calculate time difference in days (including fractional days)
            time_diff = end - start
            days_diff = time_diff.total_seconds() / (24 * 3600)  # Convert to days

            if days_diff > max_days:
                return {
                    "valid": False,
                    "error": f"Time range too large. {limits['description']}. Please select a smaller range.",
                    "max_days": max_days,
                    "current_days": round(days_diff, 2),
                }

            return {"valid": True, "days": round(days_diff, 2)}

        except ValueError as e:
            return {"valid": False, "error": f"Invalid date/time format: {str(e)}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key by making a simple test call."""
        try:
            # Get provider from config if not provided
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                
                # Make a simple test call
                _ = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5,
                )
                
                provider_name = "OpenAI"
            elif provider == "anthropic":
                from anthropic import Anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Anthropic API key is not set. Please update it in the Settings section.",
                    }
                
                client = Anthropic(api_key=api_key)
                
                # Make a simple test call
                _ = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": "Hello"}],
                )
                
                provider_name = "Anthropic"
            else:  # qwen
                from langchain_qwq import ChatQwen
                api_key = os.environ.get("DASHSCOPE_API_KEY") or self.config.get("qwen_api_key", "")
                if not api_key:
                    return {
                        "valid": False,
                        "error": "❌ Invalid API Key: The Qwen API key is not set. Please update it in the Settings section.",
                    }
                
                # Make a simple test call using LangChain
                llm = ChatQwen(model="qwen-flash", api_key=api_key)
                _ = llm.invoke([("user", "Hello")])
                
                provider_name = "Qwen"
            return {"valid": True, "message": f"{provider_name} API key is valid"}

        except Exception as e:
            error_msg = str(e)
            
            # Determine provider name for error messages
            if provider is None:
                provider = self.config.get("agent_llm_provider", "openai")
            if provider == "openai":
                provider_name = "OpenAI"
            elif provider == "anthropic":
                provider_name = "Anthropic"
            else:
                provider_name = "Qwen"

            if (
                "authentication" in error_msg.lower()
                or "invalid api key" in error_msg.lower()
                or "401" in error_msg
                or "invalid_api_key" in error_msg.lower()
            ):
                return {
                    "valid": False,
                    "error": f"❌ Invalid API Key: The {provider_name} API key is invalid or has expired. Please update it in the Settings section.",
                }
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {
                    "valid": False,
                    "error": f"⚠️ Rate Limit Exceeded: You've hit the {provider_name} API rate limit. Please wait a moment and try again.",
                }
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"💳 Billing Issue: Your {provider_name} account has insufficient credits or billing issues. Please check your {provider_name} account.",
                }
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {
                    "valid": False,
                    "error": f"🌐 Network Error: Unable to connect to {provider_name} servers. Please check your internet connection.",
                }
            else:
                return {"valid": False, "error": f"❌ API Key Error: {error_msg}"}

    def load_custom_assets(self) -> list:
        """Load custom assets from persistent JSON file."""
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            return []
        except Exception as e:
            print(f"Error loading custom assets: {e}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        """Save a custom asset symbol persistently (avoid duplicates)."""
        try:
            symbol = symbol.strip()
            if not symbol:
                return False
            if symbol in self.custom_assets:
                return True  # already present
            self.custom_assets.append(symbol)
            # write to file
            with open(self.custom_assets_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_assets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom asset '{symbol}': {e}")
            return False

    def analyze_asset(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> Dict[str, Any]:
        """Fetch data and run analysis for a given asset and timeframe."""
        df = self.fetch_finance_data(symbol, start_date, end_date, timeframe)
        results = self.run_analysis(df, symbol, timeframe)
        formatted_results = self.extract_analysis_results(results)

        if formatted_results.get("success", False):
            # Create a version without base64 images for URL encoding
            # Base64 images are too large for URL parameters
            url_safe_results = formatted_results.copy()
            url_safe_results["pattern_chart"] = ""  # Remove base64 data
            url_safe_results["trend_chart"] = ""  # Remove base64 data

            # Encode results for URL
            results_json = json.dumps(url_safe_results)
            encoded_results = urllib.parse.quote(results_json)
            redirect_url = f"/output?results={encoded_results}"

            # Store full results (with images) in session or temporary storage
            # For now, we'll pass them back in the response for the frontend to handle
            return {
                    "redirect": redirect_url,
                    "full_results": formatted_results,  # Include images in response body
                }
        else:
            return {"error": formatted_results.get("error", "Analysis failed")}
