import pytest
import torch
import unittest
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.workers.reward_manager.reward_manager import RewardManager
from verl import DataProto

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", trust_remote_code=True
        )
    return _tokenizer


def make_dummy_dataproto(prompt, response, data_source="text", rm_infos=None):
    # Minimal DataProto mock
    class DummyBatch:
        def __getitem__(self, key):
            if key == "prompts":
                return torch.tensor([prompt])
            if key == "responses":
                return torch.tensor([response])
            if key == "attention_mask":
                return torch.tensor([1] * (len(response) + len(prompt)))
            return None

    class DummyDataProto:
        batch = DummyBatch()
        non_tensor_batch = {
            "data_source": [data_source],
            "index": [0],
            "reward_model": [rm_infos or {}],
        }

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self

    return DummyDataProto()


def test_code_generation():
    config = OmegaConf.create(
        {
            "template_type": "default",
            "code_evaluator": {"max_concurrent": 1, "execute_sequential": True},
        }
    )
    tokenizer = get_tokenizer()
    reward_manager = RewardManager(config, tokenizer)
    prompt = "Write a function that adds two numbers."
    response = "<think>hello</think>```\ndef add(a, b): return a + b\n```</answer>"
    rm_infos = {
        "ground_truth": """```
def add(a, b): return a + b
```""",
        "unit_tests": [
            """
import unittest
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
"""
        ],
        "required_libs": ["pandas"],
    }
    prompt = tokenizer.encode(prompt, add_special_tokens=True)
    response = tokenizer.encode(response, add_special_tokens=True)
    data = make_dummy_dataproto(prompt, response, data_source="code", rm_infos=rm_infos)
    timing_raw = {}
    reward_tensor, extras = reward_manager.compute_rewards(data, timing_raw=timing_raw)
    print(
        "[Code Generation] Reward tensor:\n",
        reward_tensor,
        "\n[Extras]:\n",
        extras,
        sep="\n```",
    )


def test_interleaved_outline_code_unit_test():
    config = OmegaConf.create(
        {
            "template_type": "interleave",
            "code_evaluator": {
                "max_concurrent": 1,
                "execute_sequential": True,
                "auto": "http://10.128.0.30:81",
            },
        }
    )
    tokenizer = get_tokenizer()
    reward_manager = RewardManager(config, tokenizer)
    prompt = "Write a function that multiplies two numbers."
    response = (
        "<answer>Outline: Use def, multiply a and b, return result.</answer>"
        "<answer>```\ndef multiply(a, b): return a * b\n```</answer>"
        "<answer>```\nclass TestMultiply(unittest.TestCase):\n    def test_multiply(self):\n        self.assertEqual(multiply(2, 3), 6)\n```</answer>"
    )
    rm_infos = {
        "ground_truth": """```
def multiply(a, b): return a * b
```""",
        "unit_tests": [
            """
import unittest
class TestMultiply(unittest.TestCase):
    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)
"""
        ],
        "required_libs": ["pandas"],
    }
    prompt = tokenizer.encode(prompt, add_special_tokens=True)
    response = tokenizer.encode(response, add_special_tokens=True)
    data = make_dummy_dataproto(prompt, response, data_source="code", rm_infos=rm_infos)
    timing_raw = {}
    reward_tensor, extras = reward_manager.compute_rewards(data, timing_raw=timing_raw)
    print(
        """[Interleaved] Reward tensor:\n""",
        reward_tensor,
        """\n[Extras]:\n""",
        extras,
        sep="\n```",
    )


def test_natural_language_qa():
    config = OmegaConf.create(
        {
            "template_type": "default",
            "autorater_service_url": "http://10.128.0.30:81",
            "code_evaluator": {"max_concurrent": 1, "execute_sequential": True},
        }
    )
    tokenizer = get_tokenizer()
    reward_manager = RewardManager(config, tokenizer)
    prompt = "Who is the president of the United States in 2021?"
    response = "<answer>Joe Biden</answer>"
    rm_infos = {"ground_truth": "Joe Biden"}
    prompt_enc = tokenizer.encode(prompt, add_special_tokens=True)
    response_enc = tokenizer.encode(response, add_special_tokens=True)
    data = make_dummy_dataproto(
        prompt_enc, response_enc, data_source="text", rm_infos=rm_infos
    )
    timing_raw = {}
    reward_tensor, extras = reward_manager.compute_rewards(data, timing_raw=timing_raw)
    print("[QA] Reward tensor:\n", reward_tensor, "\n[Extras]:\n", extras, sep="\n```")


def test_math_qa():
    config = OmegaConf.create(
        {
            "template_type": "default",
            "autorater_service_url": "http://10.128.0.30:81",
            "code_evaluator": {"max_concurrent": 1, "execute_sequential": True},
        }
    )
    tokenizer = get_tokenizer()
    reward_manager = RewardManager(config, tokenizer)
    prompt = "What is 2 + 2?"
    response = "<answer>#### 4</answer>"
    rm_infos = {"ground_truth": "4"}
    prompt_enc = tokenizer.encode(prompt, add_special_tokens=True)
    response_enc = tokenizer.encode(response, add_special_tokens=True)
    data = make_dummy_dataproto(
        prompt_enc, response_enc, data_source="math", rm_infos=rm_infos
    )
    timing_raw = {}
    reward_tensor, extras = reward_manager.compute_rewards(data, timing_raw=timing_raw)
    print(
        "[Math] Reward tensor:\n", reward_tensor, "\n[Extras]:\n", extras, sep="\n```"
    )


def test_interleaved_qa():
    config = OmegaConf.create(
        {
            "template_type": "interleave",
            "autorater_service_url": "http://10.128.0.30:81",
            "code_evaluator": {"max_concurrent": 1, "execute_sequential": True},
        }
    )
    tokenizer = get_tokenizer()
    reward_manager = RewardManager(config, tokenizer)
    prompt = "Answer the following questions: 1) Who is the president of the United States in 2021? 2) What is the capital of France?"
    response = "<answer>Joe Biden</answer><answer>Paris</answer>"
    rm_infos = {"ground_truth": ["Joe Biden", "Paris"]}
    prompt_enc = tokenizer.encode(prompt, add_special_tokens=True)
    response_enc = tokenizer.encode(response, add_special_tokens=True)
    data = make_dummy_dataproto(
        prompt_enc, response_enc, data_source="text", rm_infos=rm_infos
    )
    timing_raw = {}
    reward_tensor, extras = reward_manager.compute_rewards(data, timing_raw=timing_raw)
    print(
        "[Interleaved QA] Reward tensor:\n",
        reward_tensor,
        "\n[Extras]:\n",
        extras,
        sep="\n```",
    )


if __name__ == "__main__":
    # test_code_generation()
    # test_interleaved_outline_code_unit_test()
    test_natural_language_qa()
    test_math_qa()
    test_interleaved_qa()
