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
    """为给定的新闻页面撰写引言部分，需遵循以下准则：
    1. 引言部分应当独立成段，简要概述文章主题。它应标明主题，建立背景，解释该主题的重要性，并总结最重要的要点，包括任何显著的争议。
    2. 引言部分应简明扼要，不超过四个结构良好的段落。
    3. 引言部分应适当引用来源。在必要处添加行内引用（例如，“美国的首都是华盛顿特区[1][3]。”）。
    """

    topic = dspy.InputField(prefix="页面的主题：", format=str)
    draft_page = dspy.InputField(prefix="草稿页面：\n", format=str)
    lead_section = dspy.OutputField(prefix="撰写引言部分：\n", format=str)


class PolishPage(dspy.Signature):
    """你是一位细心的文本编辑者，擅长找到文章中的重复信息并将其删除，以确保文章没有重复内容。你不会删除任何非重复部分，并会保留行内引用和文章结构（由“#”、“##”等表示）格式。请为以下文章执行你的工作。"""

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
