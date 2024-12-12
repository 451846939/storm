import os
import sys

from argparse import ArgumentParser
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel
from knowledge_storm.rm import YouRM, BingSearch, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG, AzureAISearch
from knowledge_storm.utils import load_api_key

def main(args):
    # 从toml文件或环境变量中加载密钥
    load_api_key(toml_file_path='secrets.toml')
    lm_configs = STORMWikiLMConfigs()

    # 设置OpenAI相关参数，根据API类型选择模型类
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

    # 初始化各个阶段使用的模型
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

    engine_args = STORMWikiRunnerArguments(
        output_dir=args.output_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )

    # 根据入参选择检索器
    match args.retriever:
        case 'bing':
            rm = BingSearch(bing_search_api=os.getenv('BING_SEARCH_API_KEY'), k=engine_args.search_top_k)
        case 'you':
            rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=engine_args.search_top_k)
        case 'brave':
            rm = BraveRM(brave_search_api_key=os.getenv('BRAVE_API_KEY'), k=engine_args.search_top_k)
        case 'duckduckgo':
            rm = DuckDuckGoSearchRM(k=engine_args.search_top_k, safe_search='On', region='us-en')
        case 'serper':
            rm = SerperRM(serper_search_api_key=os.getenv('SERPER_API_KEY'), query_params={'autocorrect': True, 'num': 10, 'page': 1})
        case 'tavily':
            rm = TavilySearchRM(tavily_search_api_key=os.getenv('TAVILY_API_KEY'), k=engine_args.search_top_k, include_raw_content=True)
        case 'searxng':
            rm = SearXNG(searxng_api_key=os.getenv('SEARXNG_API_KEY'), k=engine_args.search_top_k)
        case 'azure_ai_search':
            rm = AzureAISearch(azure_ai_search_api_key=os.getenv('AZURE_AI_SEARCH_API_KEY'), k=engine_args.search_top_k)
        case _:
            raise ValueError(f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", "searxng", or "azure_ai_search"')

    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    topic = args.topic
    runner.run(
        topic=topic,
        ground_truth_url=args.ground_truth_url,
        do_research=args.do_research,
        do_generate_outline=args.do_generate_outline,
        do_generate_article=args.do_generate_article,
        do_polish_article=args.do_polish_article,
        remove_duplicate=args.remove_duplicate,
    )
    runner.post_run()
    runner.summary()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--topic', type=str, required=False, help='The topic to research.')
    parser.add_argument('--output-dir', type=str, default='./results/gpt',
                        help='Directory to store the outputs.')
    parser.add_argument('--max-thread-num', type=int, default=3,
                        help='Maximum number of threads to use.')
    parser.add_argument('--retriever', type=str, default='bing',
                        choices=['bing', 'you', 'brave', 'serper', 'duckduckgo', 'tavily', 'searxng', 'azure_ai_search'],
                        help='The search engine API to use for retrieving information.')

    # Pipeline stage arguments
    parser.add_argument('--do-research', action='store_true',
                        help='If True, run the research stage.')
    parser.add_argument('--do-generate-outline', action='store_true',
                        help='If True, run the outline generation stage.')
    parser.add_argument('--do-generate-article', action='store_true',
                        help='If True, run the article generation stage.')
    parser.add_argument('--do-polish-article', action='store_true',
                        help='If True, run the article polishing stage.')

    # Additional parameters
    parser.add_argument('--remove-duplicate', action='store_true',
                        help='If True, remove duplicate content in the polishing stage.')
    parser.add_argument('--ground-truth-url', type=str, default='',
                        help='A ground truth URL to exclude from search results if needed.')

    # Hyperparameters
    parser.add_argument('--max-conv-turn', type=int, default=3,
                        help='Maximum number of turns in the conversation-based research.')
    parser.add_argument('--max-perspective', type=int, default=3,
                        help='Maximum number of perspectives for perspective-guided question asking.')
    parser.add_argument('--search-top-k', type=int, default=3,
                        help='Number of top search results to retrieve per query.')

    args = parser.parse_args()
    main(args)