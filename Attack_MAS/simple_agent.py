from dataclasses import dataclass, field
from typing import Callable,Any
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
import time

@dataclass
class Model:
    base_url:str
    api_key:str
    model_name:str
    temperature:float = 0.1
class SimpleAgent:
    def __init__(
        self,
        name:str,
        sys_prompt:str,
        model:Model,
        max_retry:int,
        verify_func: Callable[[str], tuple[bool, Any]],
        print_prompt:bool
    ) -> None:
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.max_retry = max_retry
        self.verify_func = verify_func
        self.print_prompt = print_prompt

    
    def generate_with_llm(
        self,
        user_prompt: str
    ) -> str:
        """_summary_
        Args:
            user_prompt (str): The user prompt.

        Raises:
            RuntimeError: Request or format error.

        Returns:
            str: The final validated result.
        """
        retry_times = self.max_retry

        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url
        )

        if self.print_prompt:
            print("=" * 50)
            print(f"{self.name} user prompt:\n{user_prompt}\n")
            print(f"{self.name} sys prompt:\n{self.sys_prompt}\n")

        last_error = None

        for attempt in range(retry_times):

            try:
                # =========================
                # LLM API request
                # =========================
                generated_text = client.chat.completions.create(
                    model=self.model.model_name,
                    messages=[
                        {"role": "system", "content": self.sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.model.temperature,
                    response_format={"type": "json_object"}
                )

                llm_res = generated_text.choices[0].message.content

                # =========================
                # Response verification
                # =========================
                success, result = self.verify_func(llm_res)

                if success:
                    return result

                last_error = result

                print(
                    f"[Agent_{self.name}]: "
                    f"Response verification failed "
                    f"(attempt {attempt + 1}/{retry_times}): {result}"
                )

            # ==========================================
            # Network / API connection errors
            # ==========================================
            except (
                APIConnectionError,
                APITimeoutError,
            ) as e:

                last_error = e

                if attempt < retry_times - 1:
                    wait_time = 2 ** attempt

                    print(
                        f"[Agent_{self.name}]: "
                        f"API request failed "
                        f"(attempt {attempt + 1}/{retry_times}): "
                        f"{type(e).__name__}: {e}"
                    )
                    print(
                        f"[Agent_{self.name}]: "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    print(
                        f"[Agent_{self.name}]: "
                        f"API request failed after "
                        f"{retry_times} attempts: "
                        f"{type(e).__name__}: {e}"
                    )

        raise RuntimeError(
            f"[Agent_{self.name}]: "
            f"LLM request/response verification failed after "
            f"{retry_times} attempts. "
            f"Last error: {last_error}"
        )

