from typing import Tuple, Optional, Union, List
import pandas as pd
from lmformatenforcer import JsonSchemaParser, CharacterLevelParser, RegexParser, StringParser
from lmformatenforcer.integrations.transformers import generate_enforced, build_token_enforcer_tokenizer_data
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

StringOrManyStrings = Union[str, List[str]]

model_id = 'google/gemma-7b-it'
device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)


def get_prompt(message: str, system_prompt: str) -> str:
    texts = [f'<bos><start_of_turn>user'
             f'{message}<end_of_turn>'
             f'<start_of_turn>model']
    # The first user input is _not_ stripped
    do_strip = False
    message = message.strip() if do_strip else message
    return message


def run(message: StringOrManyStrings,
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        num_beams: int = 1,
        required_regex: Optional[str] = None,
        required_str: Optional[str] = None,
        required_json_schema: Optional[dict] = None,
        required_json_output: Optional[bool] = None) -> Tuple[StringOrManyStrings, Optional[pd.DataFrame]]:
    is_multi_message = isinstance(message, list)
    messages = message if is_multi_message else [message]
    prompts = [get_prompt(msg, system_prompt) for msg in messages]
    inputs = tokenizer(prompts, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False,
                       padding=is_multi_message).to(device)

    generate_kwargs = dict(
        inputs,
        # streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        output_scores=True,
        return_dict_in_generate=True
    )

    parser: Optional[CharacterLevelParser] = None
    if required_regex:
        parser = RegexParser(required_regex)
    if required_str:
        parser = StringParser(required_str)
    if required_json_schema:
        parser = JsonSchemaParser(required_json_schema)
    if required_json_output:
        parser = JsonSchemaParser(None)

    if parser:
        output = generate_enforced(model, tokenizer_data, parser, **generate_kwargs)
    else:
        output = model.generate(**generate_kwargs)

    sequences = output['sequences']
    # skip_prompt=True doesn't work consistenly, so we hack around it.
    string_outputs = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in sequences]
    string_outputs = [string_output.replace(prompt[3:], '') for string_output, prompt in zip(string_outputs, prompts)]
    if parser and not is_multi_message:
        enforced_scores_dict = output.enforced_scores
        enforced_scores = pd.DataFrame(enforced_scores_dict)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.max_rows', 999)
        pd.set_option('display.float_format', ' {:,.5f}'.format)
    else:
        enforced_scores = None
    return string_outputs if is_multi_message else string_outputs[0], enforced_scores


context = '''Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American 
singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing 
and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's 
Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all 
time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a 
solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in 
Love" and "Baby Boy".'''
sent_dict = {}
for idx, sent in enumerate(sent_tokenize(context)):
    sent_dict[idx] = sent
numbered_ctx = ' '.join(str(x) + ': ' + y for x, y in sent_dict.items())

question = (f"Based on the context: {numbered_ctx}, When did Beyonce start becoming popular? "
            f"Please answer in number: format.")


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
DEFAULT_MAX_NEW_TOKENS = 5

required_format_answer_regex = r'(\d{1,2}:)'
answer_prompt = ' In number: format, When did Beyonce start becoming popular ' + required_format_answer_regex

result, enforced_scores = run(question, system_prompt=DEFAULT_SYSTEM_PROMPT, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                              required_regex=answer_prompt)
print(result)
