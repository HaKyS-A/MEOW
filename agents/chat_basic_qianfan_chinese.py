"""The chat module of ERNIE-bot series using qianfan module"""
import traceback
import qianfan
from . import global_variables


class ChatBasicQianfanChinese:
    """
    ERNIE-Bot chat module
    default module is ERNIE-Bot-8k
    """

    def __init__(self, top_p=0.75, temperature=0.9, context: list = None, model='ERNIE-Bot-8k'):
        # context = [{"role": <"user"or"assistant">, "content": ""}]
        if context is None:
            self.context = []
        else:
            self.context = context
        self.top_p = top_p
        self.temperature = temperature
        self.model = model

    def chat_basic(self, pop_fn, check_fn, **kwargs) -> int | list:
        """
        generate and return response based on the popped prompts series
        :return: error code (-1) or a list of formatted data (usually JSON)
        """
        output = []
        print('-' * 40)

        n = 0  # n is the position index of pop_fn and check_fn
        while True:
            prompt_tuple = pop_fn(n, **kwargs)  # self.context update
            prompt = prompt_tuple[0]

            if prompt == '':
                # no more prompts in prompt series
                return output

            retry = 0   # a variable to judge error

            while True:
                chat_comp = qianfan.ChatCompletion(model=self.model)
                try:
                    # LLM generation
                    response = chat_comp.do(
                        messages=self.context,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        penalty_score=1.0
                    )
                    answer = response['result']

                    # usage updates in global_variables
                    global_variables.Usage += response['usage']['total_tokens']
                    print(response['usage'])

                    # response check
                    valid, data = check_fn(answer, n)
                    if valid:
                        # pass the check
                        if data:
                            # exist output data
                            output.append(data)
                        n += 1
                        del response
                        del answer
                        break

                    # fail to pass the check
                    retry += 1
                    if retry > 5:
                        raise ValueError('Cannot generate valid output in 5 times. -00')
                    del response, answer
                except ValueError:
                    # data format error in generation
                    if retry > 5:
                        print('Cannot generate valid output in 5 times. -01')
                        return -1
                    retry += 1
                    print('LLM error')
                except Exception:
                    # Cloud service errors or other errors
                    traceback.print_exc()
                    if retry > 10:
                        print('Cloud service error')
                        return -1
                    retry += 1
                    print('Cloud service error')
