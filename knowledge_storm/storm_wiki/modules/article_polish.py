import copy
from typing import Union

import dspy

from .storm_dataclass import StormArticle
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        """
        Polish article.

        Args:
            topic (str): The topic of the article.
            draft_article (StormArticle): The draft article.
            remove_duplicate (bool): Whether to use one additional LM call to remove duplicates from the article.
        """

        article_text = draft_article.to_string()
        polish_result = self.polish_page(
            topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate
        )
        lead_section = f"# 概要\n{polish_result.lead_section}"
        polished_article = "\n\n".join([lead_section, polish_result.page])
        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        polished_article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()
        return polished_article


class WriteLeadSection(dspy.Signature):
    """为给定的新闻页面撰写引言部分，并包含引人入胜的标题和段落结构，需遵循以下准则：
     1. 引言部分：
     独立成段，概述文章主题，包括时间、地点和事件的背景。
     简明扼要，不超过四个段落，涵盖核心要点和显著争议。
     适当引用来源，在行内添加注释，例如：[1][3]。
     2. 主体内容：
     事件具体细节： 描述事件的发生经过、政策核心内容或感染情况，引用具体数据和权威机构的声明。
     背景与重要性： 提供与事件相关的历史背景，解释其社会、经济或环境影响。
     争议与反响： 收录支持者和反对者的观点，引用权威人士或组织的评论。
     3. 写作风格：
     语言生动，避免重复信息，确保段落间内容独立。
     保持条理性，融入适当的幽默或情感元素吸引读者。
     确保文本流畅自然，信息完整、准确。
     """

    topic = dspy.InputField(prefix="页面的主题：", format=str)
    draft_page = dspy.InputField(prefix="草稿页面：\n", format=str)
    lead_section = dspy.OutputField(prefix="撰写引言部分：\n", format=str)


class PolishPage(dspy.Signature):
    """你是一位细心的新闻编辑者，擅长找到文章中的重复信息并将其删除，以确保文章没有重复内容。你不会删除任何非重复部分，并会保留行内引用和文章结构（由“#”、“##”等表示）格式。请为以下文章执行你的工作。"""

    draft_page = dspy.InputField(prefix="草稿文章：\n", format=str)
    page = dspy.OutputField(prefix="修改后的文章：\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        polish_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.write_lead = dspy.Predict(WriteLeadSection)
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):
        # NOTE: Change show_guidelines to false to make the generation more robust to different LM families.
        with dspy.settings.context(lm=self.write_lead_engine, show_guidelines=False):
            lead_section = self.write_lead(
                topic=topic, draft_page=draft_page
            ).lead_section
            if "The lead section:" in lead_section:
                lead_section = lead_section.split("The lead section:")[1].strip()
        if polish_whole_page:
            # NOTE: Change show_guidelines to false to make the generation more robust to different LM families.
            with dspy.settings.context(lm=self.polish_engine, show_guidelines=False):
                page = self.polish_page(draft_page=draft_page).page
        else:
            page = draft_page

        return dspy.Prediction(lead_section=lead_section, page=page)
