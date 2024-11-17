import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Union

from .collaborative_storm_utils import clean_up_section
from ...dataclass import KnowledgeBase, KnowledgeNode


class ArticleGenerationModule(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],
        knowledge_base: KnowledgeBase,
        max_words: int = 1500,
    ):
        information = []
        cur_word_count = 0
        for index in sorted(list(all_citation_index)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            snippet = info.snippets[0]
            info_text = f"[{index}]: {snippet} (Question: {info.meta['question']}. Query: {info.meta['query']})"
            cur_snippet_length = len(info_text.split())
            if cur_snippet_length + cur_word_count > max_words:
                break
            cur_word_count += cur_snippet_length
            information.append(info_text)
        return "\n".join(information)

    def gen_section(
        self, topic: str, node: KnowledgeNode, knowledge_base: KnowledgeBase
    ):
        if node is None or len(node.content) == 0:
            return ""
        if (
            node.synthesize_output is not None
            and node.synthesize_output
            and not node.need_regenerate_synthesize_output
        ):
            return node.synthesize_output
        all_citation_index = node.collect_all_content()
        information = self._get_cited_information_string(
            all_citation_index=all_citation_index, knowledge_base=knowledge_base
        )
        with dspy.settings.context(lm=self.engine):
            synthesize_output = clean_up_section(
                self.write_section(
                    topic=topic, info=information, section=node.name
                ).output
            )
        node.synthesize_output = synthesize_output
        node.need_regenerate_synthesize_output = False
        return node.synthesize_output

    def forward(self, knowledge_base: KnowledgeBase):
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_paragraph = {}

        # Define a function to generate paragraphs for nodes
        def _node_generate_paragraph(node):
            node_gen_paragraph = self.gen_section(
                topic=knowledge_base.topic, node=node, knowledge_base=knowledge_base
            )
            lines = node_gen_paragraph.split("\n")
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:
                lines = lines[1:]
            node_gen_paragraph = "\n".join(lines)
            path = " -> ".join(node.get_path_from_root())
            return path, node_gen_paragraph

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node
                for node in all_nodes
            }

            # Collect the results as they complete
            for future in as_completed(future_to_node):
                path, node_gen_paragraph = future.result()
                node_to_paragraph[path] = node_gen_paragraph

        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                hash_tag = "#" * level + " "
                cur_path = " -> ".join(cur_root.get_path_from_root())
                node_gen_paragraph = node_to_paragraph[cur_path]
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")
                for child in cur_root.children:
                    to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        for child in knowledge_base.root.children:
            to_return.extend(helper(child, level=1))

        return "\n".join(to_return)


class WriteSection(dspy.Signature):
    """根据收集到的信息撰写维基百科的一个章节。你将获得主题、需要撰写的章节名称和相关信息。
    每条信息将包含原始内容，以及与该信息相关的问题和查询。
    请按照以下格式撰写：
    在句中使用[1]、[2]、...、[n]的形式（例如，“美国的首都是华盛顿特区[1][3]。”）。你不需要在结尾添加参考或来源部分。
    """

    info = dspy.InputField(prefix="收集到的信息：\n", format=str)
    topic = dspy.InputField(prefix="页面的主题：", format=str)
    section = dspy.InputField(prefix="需要撰写的章节：", format=str)
    output = dspy.OutputField(
        prefix="撰写章节内容并正确使用行内引用（开始撰写，不包括页面标题、章节名称，也不要尝试撰写其他章节。不要以主题名称开头）：\n",
        format=str,
    )
