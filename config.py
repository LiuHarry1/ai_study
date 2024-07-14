import os.path
from dataclasses import dataclass
from pathlib import Path

@dataclass
class nlp_tc_config:
    ROOT_PATH = str(Path(__file__).parent)
    WORD_2_VECTOR =os.path.join(ROOT_PATH, 'ai_learn',  'word2vector')
    GLOVE_PATH = os.path.join(WORD_2_VECTOR, 'glove')


if __name__ == '__main__':
    print(nlp_tc_config.ROOT_PATH)
    print(nlp_tc_config.DATASET)
    print(nlp_tc_config.TEXT_CNN)
    print(nlp_tc_config.GLOVE_PATH)


