"""
STORM Wiki pipeline powered by GPT-3.5/4 and You.com search engine.
You need to set up the following environment variables to run this script:
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_API_TYPE: OpenAI API type (e.g., 'openai' or 'azure')
    - AZURE_API_BASE: Azure API base URL if using Azure API
    - AZURE_API_VERSION: Azure API version if using Azure API
    - YDC_API_KEY: You.com API key; BING_SEARCH_API_KEY: Bing Search API key, SERPER_API_KEY: Serper API key, BRAVE_API_KEY: Brave API key, or TAVILY_API_KEY: Tavily API key

Output will be structured as below
args.output_dir/
    topic_name/  # topic_name will follow convention of underscore-connected topic name w/o space and slash
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Raw search results from search engine
        direct_gen_outline.txt          # Outline directly generated with LLM's parametric knowledge
        storm_gen_outline.txt           # Outline refined with collected information
        url_to_info.json                # Sources that are used in the final article
        storm_gen_article.txt           # Final article generated
        storm_gen_article_polished.txt  # Polished final article (if args.do_polish_article is True)
"""

import os

from flask import Flask, request, jsonify

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import YouRM, BingSearch, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG, AzureAISearch
from knowledge_storm.utils import load_api_key

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run():
    # 从请求中提取参数
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request. JSON body is required."}), 400

    try:
        # 加载 API 密钥
        load_api_key(toml_file_path='secrets.toml')
        lm_configs = STORMWikiLMConfigs()

        # 设置 OpenAI 相关参数
        openai_kwargs = {
            'api_key': os.getenv("OPENAI_API_KEY"),
            'temperature': 1.0,
            'top_p': 0.9,
        }

        ModelClass = OpenAIModel if os.getenv('OPENAI_API_TYPE') == 'openai' else AzureOpenAIModel
        gpt_35_model_name = 'gpt-3.5-turbo' if os.getenv('OPENAI_API_TYPE') == 'openai' else 'gpt-35-turbo'
        gpt_4_model_name = 'gpt-4o'

        if os.getenv('OPENAI_API_TYPE') == 'azure':
            openai_kwargs['api_base'] = os.getenv('AZURE_API_BASE')
            openai_kwargs['api_version'] = os.getenv('AZURE_API_VERSION')

        # 初始化模型
        conv_simulator_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
        question_asker_lm = ModelClass(model=gpt_35_model_name, max_tokens=500, **openai_kwargs)
        outline_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=400, **openai_kwargs)
        article_gen_lm = ModelClass(model=gpt_4_model_name, max_tokens=700, **openai_kwargs)
        article_polish_lm = ModelClass(model=gpt_4_model_name, max_tokens=4000, **openai_kwargs)

        lm_configs.set_conv_simulator_lm(conv_simulator_lm)
        lm_configs.set_question_asker_lm(question_asker_lm)
        lm_configs.set_outline_gen_lm(outline_gen_lm)
        lm_configs.set_article_gen_lm(article_gen_lm)
        lm_configs.set_article_polish_lm(article_polish_lm)

        # 设置引擎参数
        engine_args = STORMWikiRunnerArguments(
            output_dir=data.get('output_dir', './results/gpt'),
            max_conv_turn=data.get('max_conv_turn', 3),
            max_perspective=data.get('max_perspective', 3),
            search_top_k=data.get('search_top_k', 3),
            max_thread_num=data.get('max_thread_num', 3),
        )

        # 选择检索器
        # 从 HTTP 参数中获取检索器及其 key
        retriever = data.get('retriever', 'you')
        retriever_api_key = data.get('retriever_api_key')
        if not retriever_api_key:
            return jsonify({"error": "Missing retriever_api_key in the request."}), 400

        # 选择检索器
        match retriever:
            case 'bing':
                rm = BingSearch(bing_search_api=retriever_api_key, k=engine_args.search_top_k)
            case 'you':
                rm = YouRM(ydc_api_key=retriever_api_key, k=engine_args.search_top_k)
            case 'brave':
                rm = BraveRM(brave_search_api_key=retriever_api_key, k=engine_args.search_top_k)
            case 'duckduckgo':
                rm = DuckDuckGoSearchRM(k=engine_args.search_top_k, safe_search='On', region='us-en')
            case 'serper':
                rm = SerperRM(serper_search_api_key=retriever_api_key, query_params={'autocorrect': True, 'num': 10, 'page': 1})
            case 'tavily':
                rm = TavilySearchRM(tavily_search_api_key=retriever_api_key, k=engine_args.search_top_k, include_raw_content=True)
            case 'searxng':
                rm = SearXNG(searxng_api_key=retriever_api_key, k=engine_args.search_top_k)
            case 'azure_ai_search':
                rm = AzureAISearch(azure_ai_search_api_key=retriever_api_key, k=engine_args.search_top_k)
            case _:
                return jsonify({"error": f"Invalid retriever: {retriever}"}), 400


        # 初始化运行器
        runner = STORMWikiRunner(engine_args, lm_configs, rm)

        # 运行流程
        runner.run(
            topic=data.get('topic', ''),
            ground_truth_url=data.get('ground_truth_url', ''),
            do_research=data.get('do_research', False),
            do_generate_outline=data.get('do_generate_outline', False),
            do_generate_article=data.get('do_generate_article', False),
            do_polish_article=data.get('do_polish_article', False),
            remove_duplicate=data.get('remove_duplicate', False),
        )
        runner.post_run()
        runner.summary()

        return jsonify({"status": "success"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)