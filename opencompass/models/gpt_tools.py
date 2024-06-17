import json
import os
import uuid
import subprocess
from typing import Dict, List, Optional, Union
from pathlib import Path

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class GptTool(BaseAPIModel):
    """Model wrapper around DeepseekAPI.

    Documentation:

    Args:
        path (str): The name of DeepseekAPI model.
            e.g. `gpt-4o`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
            self,
            path: str,
            key: str,
            query_per_second: int = 2,
            max_seq_len: int = 2048,
            meta_template: Optional[Dict] = None,
            retry: int = 2,
            num_workers: int = 8,
            system_prompt: str = '',
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)

        gpt_tool = os.environ.get("GPT_TOOL")
        if not gpt_tool:
            raise RuntimeError("GPT_TOOL is not set")

        self.cmd = [
            str(gpt_tool),
            '-key', key,
            '-input', '',
            '-output', '',
            '-model', str(path),
            '-promptTag', 'question',
            '-answerTag', 'answer',
            '-worker', str(num_workers),
            '-topP', '0',
            '-topK', '1',
            '-temperature', '0.0001',
        ]

        self.model = path
        self.system_prompt = system_prompt

    def generate(
            self,
            inputs: List[PromptType],
            max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        cmd = self.cmd.copy()
        cache_dir = Path(os.environ.get("TMP", '/tmp')) / 'gpt4'
        cache_dir.mkdir(exist_ok=True)
        uid = uuid.uuid4()

        questions_file = cache_dir / f"questions-{uid}.jsonl"
        answers_file = cache_dir / f"answers-{uid}.jsonl"

        cmd[4] = str(questions_file)
        cmd[6] = str(answers_file)

        with open(questions_file, 'w') as f:
            for i, input in enumerate(inputs):
                assert isinstance(input, (str, PromptList))
                if isinstance(input, str):
                    messages = input

                else:
                    messages = []
                    for item in input:
                        messages.append(item['prompt'])
                    messages = '\n'.join(messages)
                f.write(json.dumps({'id': i, 'question': str(messages)}, ensure_ascii=False) + '\n')

        results = []
        subprocess.run(cmd, check=True)

        with open(answers_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                results.append((result['id'], result['answer']))

        results = sorted(results, key=lambda x: x[0])
        results = [answer for _, answer in results]
        self.flush()
        return results
