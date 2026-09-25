from __future__ import annotations

import json

import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple



from openai import OpenAI

@dataclass
class ShortMemorItem:
    current_turn:int
    speaker_c:str
    speaker_s:str
    embedding:List[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_turn": self.current_turn,
            "speaker_c": self.speaker_c,
            "speaker_s": self.speaker_s,
        }
    
class STM:
    def __init__(
    self,
    enable_embedding_retrieve:bool = False,
    embedding_model: str = "text-embedding-v3",
    embedding_dim: int = 256,
    api_key: Optional[str] = "your_api",
    base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.enable_embedding_retrieve = enable_embedding_retrieve
        self._store:list[ShortMemorItem] = []
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "base_url",
        )
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

    def embed(self, content: str) -> List[float]:
        completion = self.client.embeddings.create(
            model=self.embedding_model,
            input=content,
            dimensions=self.embedding_dim,
            encoding_format="float",
        )
        data = json.loads(completion.model_dump_json())
        return data["data"][0]["embedding"]

    def get_all_STM_for_prompt(self,print_stm_promt:bool = False) -> str:
        self._store.sort(key=lambda item: item.current_turn)
        prompt = ""
        for item in self._store:
            item_prompt = f"""
Turn_{item.current_turn}:
Customer:{item.speaker_c}
Target Customer Service Agent:{item.speaker_s}"""
            prompt = prompt + item_prompt
        if (print_stm_promt):
            print(prompt)
        return prompt

    def add(
        self,
        current_turn: int,
        speaker_c: str,
        speaker_s: str,
    ) -> None:
        content = speaker_c + speaker_s
        if(self.enable_embedding_retrieve):
            embedding = self.embed(content)
            self._store.append(
                ShortMemorItem(
                    current_turn=current_turn,
                    speaker_c=speaker_c,
                    speaker_s=speaker_s,
                    embedding=embedding
                )
            )
        else:
            self._store.append(
                ShortMemorItem(
                    current_turn=current_turn,
                    speaker_c=speaker_c,
                    speaker_s=speaker_s
                )
            )
        print("[STM]: Added")

    def clear(self):
        self._store.clear()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[ShortMemorItem, float]]:
        pass

    def retrieve(
        self,
        query: str | None = None,
        top_k: int | None = 5,
        metadata_filter: Optional[Dict[str, str]] = None,
        **_: Any,
    ) -> List[ShortMemorItem]:
        if not query:
            return []
        q_emb = self.embed(query)
        print("[STM] Retrieve by embedding")
        return [
            it
            for it, _ in self.search(
                q_emb, top_k=top_k or 5, metadata_filter=metadata_filter
            )
        ]


    def get_STM_retrieve_result_in_useable_form(self,print_stm_promt: bool = False) -> str:
        if self.enable_embedding_retrieve:
            return ""
        else:
            return self.get_all_STM_for_prompt(print_stm_promt=print_stm_promt)