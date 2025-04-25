import numpy as np
import openai
from typing import List, Optional
# from loguru import logger
from ...rl.basic_agent import Basic_Agent

# logger.add("logs/OPRO_{time}.log")
class LLMAgent:
    # use func 'run' call client to get response and parse the output to theta vector. args: meta_prompt
    def __init__(self,API_KEY) -> None:
        self.api_key = API_KEY
        # self.client = openai.Client(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.client = None
        self.model = "deepseek-chat"  # or any other model you want to use
        self.temperature = 1.3 # please refer to https://api-docs.deepseek.com/quick_start/parameter_settings for suitable settings
        self.max_tokens = 1024

    def reconnect(self):
        """
        Reconnect to the OpenAI API.
        """
        self.client = openai.Client(api_key=self.api_key, base_url="https://api.deepseek.com")
        # logger.info("Reconnected to OpenAI API.")

    def close(self):
        """
        Close the connection to the OpenAI API.
        """
        if self.client:
            self.client.close()
        #     logger.info("Closed connection to OpenAI API.")
        # else:
        #     logger.warning("No active connection to close.")


    def parse_model_output_for_theta(self, model_output: str) -> Optional[List[float]]:
        """
        Parse the model's output to extract the theta vector.

        Args:
            model_output (str): The raw text output from the language model.

        Returns:
            List[float] or None: Extracted theta vector if found, otherwise None.
        """
        import re
        pattern = r"\[([^\[\]]+?)\]"

        matches = re.findall(pattern, model_output)
        last_valid_values = None

        for match in matches:
            try:
                values = [float(x.strip()) for x in match.split(',')]
                last_valid_values = values  # update each time we find a valid one
            except ValueError:
                continue

        return last_valid_values  # will be None if none matched


    def process_output(self, output):
        res = []
        for s in output:
            parsed_theta = self.parse_model_output_for_theta(s)
            if parsed_theta is not None:
                res.append(parsed_theta)
        return res

    def call_client(self, messages, n=1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
        )
        collect_resp = []
        for i in range(n):
            collect_resp.append(response.choices[i].message.content)
        return collect_resp

    def run(self, meta_prompt, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": meta_prompt,
            },
        ]

        # Call the client with the messages
        output = self.call_client(messages, n=batch_size)
        # logger.info(f"Model output: {output[0]}")
        # Process the output
        res = self.process_output(output)
        # logger.info(f"Processed output: {res}")
        return res

class OPRO(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.llm_agent = LLMAgent(config.api_key)
        self.max_episodes = 10
        self.max_num_pairs = 10


    def __str__(self):
        return "OPRO"

    def train_episode(self,
                      envs,
                      seeds):
        raise NotImplementedError("OPRO does not support train_episode method.")

    def train_epoch(self):
        raise NotImplementedError("OPRO does not support train_epoch method.")

    def gen_meta_prompt_multi(
            self,
            old_value_pairs_set,
            num_input_decimals=5,
            num_output_decimals=5,
            max_num_pairs=100,
    ):
        """Generate the meta-prompt for general d-dimensional optimization.

        Args:
        old_value_pairs_set (set): set of (theta, value) pairs, where theta is a list.
        num_input_decimals (int): decimals for theta values.
        num_output_decimals (int): decimals for function value.
        max_num_pairs (int): number of examples to include.

        Returns:
        meta_prompt (str)
        """
        def round_list(lst, num_decimals): # 把list中元素保留num_decimals位小数
            return [
                round(x, num_decimals) if num_decimals > 0 else int(x)
                for x in lst
            ]

        rounded_pairs = set(
            (
                tuple(round_list(theta, num_input_decimals)),
                round(value, num_output_decimals) if num_output_decimals > 0 else int(value),
            )
            for theta, value in old_value_pairs_set
        )

        sorted_pairs = sorted(rounded_pairs, key=lambda x: -x[1])[-max_num_pairs:]

        examples = ""
        for theta, value in sorted_pairs:
            examples += f"\ninput: theta={list(theta)}\nvalue: {value}\n"

        meta_prompt = (
            "Now you will help me minimize a function with multiple input variables "
            f"theta = [θ₁, θ₂, ..., θ_d]. I have some theta vectors and the corresponding "
            "function values. The goal is to find a theta with a smaller function value than any seen before."
        )
        meta_prompt += "\n\n" + examples.strip()
        meta_prompt += (
            "\n\nGive me a new theta vector that is different from all vectors above, "
            "and has a function value lower than any of the above. Do not write code. "
            "The output must end with a vector like [θ₁, θ₂, ..., θ_d], where all θ values are numerical."
        )

        return meta_prompt

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        self.llm_agent.reconnect()
        results = None

        # Initialize old_value_pairs with the initial population
        old_value_pairs = env.reset()
        dim = env.problem.dim

        for ep in range(self.max_episodes):
            # Generate the meta-prompt
            meta_prompt = self.gen_meta_prompt_multi(old_value_pairs_set = old_value_pairs, max_num_pairs=self.max_num_pairs)
            # logger.info(f"Meta prompt: {meta_prompt}")
            # logger.info(f"Episode {ep + 1}/{self.max_episodes} - rollouting")


            # Call the LLM agent to get the new theta
            new_thetas = self.llm_agent.run(meta_prompt)
            # logger.info(f"New thetas: {new_thetas}")
            old_value_pairs, _, _ = env.step(new_thetas)

        self.llm_agent.close()
        # Collect results
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')

        results = {'cost': env_cost, 'fes': env_fes}

        if self.config.full_meta_data:
            meta_X = env.get_env_attr('meta_X')
            meta_Cost = env.get_env_attr('meta_Cost')
            metadata = {'X': meta_X, 'Cost': meta_Cost}
            results['metadata'] = metadata
        for key in required_info.keys():
            results[key] = getattr(env, required_info[key])
        
        # logger.info(f"Results: {results}")
        return results


