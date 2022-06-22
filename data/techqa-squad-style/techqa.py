# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""TechQA: SQuAD style version of TechQA."""


import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive


logger = datasets.logging.get_logger(__name__)


_URL = "https://github.com/saptarshi059/generalization-hypothesis/tree/main/data/techqa-squad-style"
_URLS = {
    "train": _URL + "training_Q_A_context.json",
    "dev": _URL + "dev_Q_A_context.json",
}


class TechQAConfig(datasets.BuilderConfig):
    """BuilderConfig for TechQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for TechQA.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TechQAConfig, self).__init__(**kwargs)


class TechQA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        TechQAConfig(
            name="techqa-squad-style",
            version=datasets.Version("1.0.0", ""),
            description="techqa-squad-style",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        '''
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]
        '''

        train_url = 'https://github.com/saptarshi059/generalization-hypothesis/blob/main/data/techqa-squad-style/training_Q_A_context.json'
        dev_url = 'https://github.com/saptarshi059/generalization-hypothesis/blob/main/data/techqa-squad-style/dev_Q_A_context.json'
        
        auth = ('saptarshi059', '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')

        r = requests.get(url, auth=auth)

        os.mkdir('my_temp')
        
        with open('my_temp/training_Q_A_context.json', 'w') as f:
            json.dump(r.json(), f)


        with open('my_temp/dev_Q_A_context.json', 'w') as f:
            json.dump(r.json(), f)

        #url = _URLs[self.config.name]
        #downloaded_filepath = dl_manager.download_and_extract(r)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": 'my_temp/training_Q_A_context.json'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": 'my_temp/dev_Q_A_context.json'},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            techqa = json.load(f)
            for article in techqa:
                title = article.get('QUESTION_TITLE', "")
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                if article['ANSWERABLE'] == 'Y':
                    answer_starts = [article['START_OFFSET']]
                    answers = [article['ANSWER']]
                else:
                    answer_starts = []
                    answers = []
                # Features currently used are "context", "question", and "answers".
                # Others are extracted here for the ease of future expansions.
                yield key, {
                    "title": title,
                    "context": context,
                    "question": article['QUESTION_TEXT'],
                    "id": article['QUESTION_ID'],
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers,
                    },
                }
                key += 1
