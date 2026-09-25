# -*- coding: utf-8 -*-
"""ReAct-style agent with 6 memory/context management tools (AgeMem)."""
from __future__ import annotations
import asyncio
import httpx
from openai import APIConnectionError
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
from dataclasses import dataclass

import shortuuid
from pydantic import BaseModel, ValidationError

from agentscope.agent import AgentBase
from agentscope._logging import logger
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase

from agentscope.message import Msg, TextBlock, ToolResultBlock, ToolUseBlock
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit, ToolResponse, execute_python_code

from .memory import AgentScopeLongtermMemory,MemoryItem
from .prompts import SUMMARY_CONTEXT_SYS_PROMPT, TEXT_SIMILARITY_SYS_PROMPT
from .src import (
    GenerateResponseSchema,
    chat_client,
    extract_score,
    finish_function_pre_print_hook,
)

from AgeMem_code_agentscope.file_tools import append_json_array_to_jsonl,append_json_to_jsonl

# Record task status
@dataclass
class TaskState:
    task_id: str
    current_stage: str # initial_user_query、cross_session_queries、cross_task_generalization_queries
    current_query:int
    current_turn: int 


class AgeMem(AgentBase):
    """ReAct-style agent with AgeMem (6 tools)."""

    finish_function_name: str = "generate_response"
    """The function name used to finish replying and return a response to
    the user."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        task_state: TaskState,
        memory_log_dir: str = None,
        toolkit: Optional[Toolkit] = None,
        memory: Optional[MemoryBase] = None,
        max_iters: int = 10,
        max_context_tokens: int = 32768,
        auto_summary_token_threshold: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Static variables in the agent
        self.name = name
        self._sys_prompt = sys_prompt
        self.model = model
        self.formatter = formatter
        self.max_iters = max_iters
        self.max_context_tokens = max_context_tokens
        self.auto_summary_token_threshold = auto_summary_token_threshold

        self.chat_client = kwargs.get("chat_client", chat_client(api_key="", base_url=""))
        # -------------- Task State --------------
        self.task_state = task_state

        # -------------- Memory management --------------
        # Record the Long term memory
        self.memory_log_dir = memory_log_dir
        self.memory_manager = memory or AgentScopeLongtermMemory(
            embedding_model="text-embedding-v3",
            embedding_dim=256,
        )

        # memory tools management 
        self.memory_tool_names = {
            # short memory
            "summary_context",
            "filter_context",
            # long memory
            "retrieve_memory",
            "add_memory",
            "update_memory",
            "delete_memory",
        }
        self.enabled_memory_tools = set(self.memory_tool_names)


        # Record the Short term memory
        self.context_messages: List[Msg] = []

        # Stage tracking for multi-stage training
        self.current_stage: int = 0

        # -------------- Tool management --------------
        # If None, a default Toolkit will be created
        self.toolkit = toolkit or Toolkit()
        self.toolkit.register_tool_function(getattr(self, self.finish_function_name))
        self._required_structured_model: Optional[Type[BaseModel]] = None
        self.toolkit.set_extended_model(self.finish_function_name, GenerateResponseSchema)
        self.toolkit.register_tool_function(self.summary_context)
        self.toolkit.register_tool_function(self.filter_context)
        self.toolkit.register_tool_function(self.retrieve_memory)
        self.toolkit.register_tool_function(self.add_memory)
        self.toolkit.register_tool_function(self.update_memory)
        self.toolkit.register_tool_function(self.delete_memory)

        self.toolkit.register_tool_function(execute_python_code)

        self.register_state("name")
        self.register_state("sys_prompt")
        self.register_instance_hook(
            "pre_print",
            "finish_function_pre_print_hook",
            finish_function_pre_print_hook,
        )

    @property
    def sys_prompt(self) -> str:
        return self._sys_prompt

    async def _call_model_with_retry(self, prompt):
        max_retries = 3

        for retry_idx in range(max_retries):
            try:
                response = await self.model(
                    prompt,
                    tools=self._get_active_tool_schemas(),
                )

                if not hasattr(response, "content"):
                    final_response = None

                    async for chunk in response:
                        final_response = chunk

                    response = final_response

                    if response is None:
                        raise RuntimeError(
                            "Model stream yielded no response."
                        )

                return response

            except (
                APIConnectionError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.TimeoutException
            ) as e:

                if retry_idx == max_retries - 1:
                    raise

                wait_time = 2 ** retry_idx

                print(
                    f"Model request failed "
                    f"(attempt {retry_idx + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )

                await asyncio.sleep(wait_time)

    def _append_context(self, name: str, content: str, role: str) -> None:
        self.context_messages.append(
            Msg(name=name or role, content=content, role=role)
        )

    async def reply(
        self,
        msg: Optional[Msg | List[Msg]] = None,
        structured_model: Optional[Type[BaseModel]] = None,
        clear_context: bool = False,
    ) -> Msg:
        """Reply to the message."""

        # Clear the context if clear_context is True
        if clear_context:
            self.context_messages.clear()
        if msg is not None:
            if isinstance(msg, Msg):
                msg = [msg]
            for m in msg:
                self.context_messages.append(m)
            # await self._retrieve_from_long_term_memory(msg)

        self._required_structured_model = structured_model
        if structured_model:
            self.toolkit.set_extended_model(
                self.finish_function_name,
                structured_model,
            )

        reply_msg = None
        for _ in range(self.max_iters):
            prompt = await self.formatter.format(
                msgs=[
                    Msg("system", self.sys_prompt, "system"),
                    *self.context_messages,
                ],
            )
            logger.debug(f"User message {_+1}: {prompt}")
            logger.debug(f"Agent prompt length: {len(prompt)}")     
            # Retry Res -Begin
            response = await self._call_model_with_retry(prompt)
            # Retry Res -End
            msg = Msg(
                name=self.name,
                content=list(response.content),
                role="assistant",
            )
            if msg and not msg.has_content_blocks("tool_use"):
                msg = Msg.from_dict(msg.to_dict())
                msg.content = [
                    ToolUseBlock(
                        id=shortuuid.uuid(),
                        type="tool_use",
                        name=self.finish_function_name,
                        input={"response": msg.get_text_content()},
                    ),
                ]

            self.context_messages.append(msg)

            # print(f"msg: {msg}")

            futures = [
                self._apply_tool(tool_call)
                for tool_call in msg.get_content_blocks("tool_use")
            ]
            acting_responses = [await f for f in futures]
            for acting_msg in acting_responses:
                reply_msg = reply_msg or acting_msg
            if reply_msg:
                break

        if reply_msg is None:
            reply_msg = await self._summarizing()
        self.context_messages.append(reply_msg)
        return reply_msg

    async def _apply_tool(self, tool_call: ToolUseBlock) -> Msg | None:
        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )
        try:
            # Block disabled tools.
            tool_name = tool_call["name"]

            if (
                tool_name in self.memory_tool_names
                and tool_name not in self.enabled_memory_tools
            ):
                print(f"Tool {tool_name} is currently disabled.")
                tool_res_msg.content[0]["output"] = [
                    TextBlock(
                        type="text",
                        text=f"Tool {tool_name} is currently disabled.",
                    ),
                ]
                return None
            
            tool_res = await self.toolkit.call_tool_function(tool_call)
            response_msg = None
            async for chunk in tool_res:
                tool_res_msg.content[0]["output"] = chunk.content
                if chunk.metadata and chunk.metadata.get("success", True):
                    if tool_call["name"] == self.finish_function_name:
                        response_msg = chunk.metadata.get("response_msg")
            return response_msg
        finally:
            self.context_messages.append(tool_res_msg)

    def generate_response(self, response: str, **kwargs: Any) -> ToolResponse:
        """Final response to the user."""
        response = response or kwargs.get("reply", "")
        response_msg = Msg(self.name, response, "assistant")

        # Structured output support (optional)
        if self._required_structured_model:
            try:
                structured_data = dict(kwargs)
                if hasattr(self._required_structured_model, "model_fields"):
                    fields = self._required_structured_model.model_fields
                    if "result" in fields and "result" not in structured_data:
                        structured_data["result"] = response
                    if "response" in fields and "response" not in structured_data:
                        structured_data["response"] = response
                response_msg.metadata = (
                    self._required_structured_model.model_validate(structured_data).model_dump()
                )
            except ValidationError as e:
                return ToolResponse(
                    content=[TextBlock(type="text", text=f"Arguments Validation Error: {e}")],
                    metadata={"success": False, "response_msg": None},
                )
        return ToolResponse(
            content=[TextBlock(type="text", text="Successfully generated response.")],
            metadata={"success": True, "response_msg": response_msg},
            is_last=True,
        )

    # ---------- AgeMem Tool implementations (aligned return messages) ----------
    async def summary_context(
        self,
        span
    ) -> ToolResponse:
        """Summarize selected rounds in the current context to reduce tokens while preserving key info.

        Args:
            span (`str`):
                The range to summarize. "all" | integer string (last N).

        Returns:
            `ToolResponse`:
                The response containing the result of summarizing the context.
        """
        context_messages = self.context_messages[:]
        last_idx = len(context_messages) - 1
        # The message that invoked this tool (assistant with summary_context tool_calls); must stay at end.
        keeper = context_messages[last_idx] if context_messages else None

        filtered: List[Msg] = []
        non_system_messages: List[tuple[int, Msg]] = []
        for i, m in enumerate(context_messages):
            if m.role == "system":
                filtered.append(m)
            else:
                non_system_messages.append((i, m))

        messages_to_summarize: List[tuple] = []
        indices_to_replace: List[int] = []
        if span == "all":
            messages_to_summarize = list(non_system_messages)
            indices_to_replace = [i for i, _ in non_system_messages]
        else:
            try:
                n = int(span)
                messages_to_summarize = non_system_messages[-n:]
                indices_to_replace = [i for i, _ in messages_to_summarize]
            except Exception:
                messages_to_summarize = list(non_system_messages)
                indices_to_replace = [i for i, _ in non_system_messages]

        # Never remove the last message (assistant that called summary_context), so after append we have assistant + tool.
        indices_to_replace = [i for i in indices_to_replace if i != last_idx]

        if not messages_to_summarize:
            return ToolResponse(
                content=[TextBlock(type="text", text="The span is invalid. There are no messages to summarize.")],
                metadata={"success": False, "span": span},
            )

        conversation_text = "\n".join(
            f"{m.role}: {m.get_text_content()}" for _, m in messages_to_summarize
        )
        summary = self.chat_client.chat(
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_CONTEXT_SYS_PROMPT.format(conversation_text=conversation_text),
                }
            ],
            model_name="qwen-max",
        )
        note = f"Successfully summarized the context of {len(messages_to_summarize)} messages. \n Summary: {summary}"

        if span == "all":
            self.context_messages = [keeper] if keeper else []
        else:
            if not indices_to_replace:
                self.context_messages = context_messages
            else:
                lo, hi = min(indices_to_replace), max(indices_to_replace)
                indices_to_remove = set(indices_to_replace)
                for i in range(lo, hi + 1):
                    if context_messages[i].role == "system":
                        indices_to_remove.add(i)
                indices_to_remove.discard(last_idx)
                filtered = [m for i, m in enumerate(context_messages) if i not in indices_to_remove]
                self.context_messages = filtered

        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "summary_msg": summary, "span": span},
        )

    async def filter_context(
        self, 
        criteria: str
        ) -> ToolResponse:
        """Filters out irrelevant or outdated content from the conversation context to improve task-solving efficiency. 

        Args:
            criteria (`str`):
                The criteria for content removal. Can be keywords, phrases, or descriptions of content types to remove (e.g., ’the birthday of John’, ’the age of Mary’).

        Returns:
            `ToolResponse`:
                The response containing the result of clearing the context.
        """
        filtered = []
        removed_count = 0

        for m in self.context_messages:
            if m.role == "system":
                filtered.append(m)
                continue
            if criteria:
                # Use LLM-based similarity matching for better accuracy
                similarity_text = self.chat_client.chat(
                    messages=[{
                        "role": "user",
                        "content": TEXT_SIMILARITY_SYS_PROMPT.format(
                            text1=criteria,
                            text2=m.get_text_content(),
                        ),
                    }],
                    model_name="qwen-max",
                )
                if extract_score(similarity_text, default=0.0) >= 0.1:
                    removed_count += 1
                    continue
            filtered.append(m)

        self.context_messages = filtered
        note = f"Successfully cleared {removed_count} messages from the context."
        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "removed_count": removed_count, "criteria": criteria},
        )

    async def retrieve_memory(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None,
    ) -> ToolResponse:
        """Retrieve relevant memories from the long term memory store and attach them to context.

        Args:
            query (`str`):
                The search query to find relevant memories. Should describe what kind of information or context is needed.
            top_k (`int`, defaults to `3`):
                The maximum number of memories to retrieve. Defaults to 3.
            metadata_filter (`dict`, defaults to `None`):
                Optional metadata constraints.

        Returns:
            `ToolResponse`:
                The response containing the result of retrieving the memories.
        """
        items = await self.memory_manager.retrieve(
            query, int(top_k), metadata_filter or {}
        )
        if items:
            block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)
            text = f"Successfully retrieved {len(items)} memories. Detailed memories: {block}"
            # Log the retrieval results.
            if self.memory_log_dir != None and self.memory_log_dir != "":
                log_items = []
                for it in items:
                    temp_it = it.to_dict_no_embedding()
                    temp_it["memory_action_type"] = "retrieve"
                    temp_it["task_id"] = self.task_state.task_id
                    temp_it["current_stage"] = self.task_state.current_stage
                    temp_it["current_query"] = self.task_state.current_query
                    temp_it["current_turn"] = self.task_state.current_turn
                    log_items.append(temp_it)
                append_json_array_to_jsonl(log_items, self.memory_log_dir)
        else:
            if self.memory_log_dir != None and self.memory_log_dir != "":
                log_item = {}
                log_item["memory_id"] = "None"
                log_item["content"] = "None"
                log_item["metadata"] = "None"
                log_item["memory_action_type"] = "retrieve"
                log_item["task_id"] = self.task_state.task_id
                log_item["current_stage"] = self.task_state.current_stage
                log_item["current_query"] = self.task_state.current_query
                log_item["current_turn"] = self.task_state.current_turn
                append_json_to_jsonl(self.memory_log_dir,log_item)

            text = "No related memories found."
        
        return ToolResponse(
            content=[TextBlock(type="text", text=text)],
            metadata={"success": True, "query": query},
        )

    async def add_memory(
        self,
        content: str,
        metadata: dict = None,
        memory_type: str = "general",
    ) -> ToolResponse:
        """Add new information into the long term memory store for future retrieval,avoid storing personally identifiable information such as customer names, email addresses, and phone numbers.

        Args:
            content (`str`):
                The content to store.
            metadata (`dict`, defaults to `None`):
                Optional metadata tags.
            memory_type (`str`, defaults to `"general"`):
                Logical type tag; will be stored as metadata["type"].

        Returns:
            `ToolResponse`:
                The response containing the result of adding the memory.
        """
        md = dict(metadata or {})
        if memory_type:
            md["type"] = memory_type

        # Add stage information to metadata
        md["stage"] = str(self.current_stage)
        mem_id = str(uuid.uuid4())
        await self.memory_manager.add(mem_id, content, md)
        note = f"Successfully added a new memory. Memory ID: {mem_id}, content: {content}"
        # log
        if self.memory_log_dir != None and self.memory_log_dir != "":
            added_memory_item_for_log = MemoryItem(memory_id=mem_id,content=content,metadata=md)
            memory_item_for_log = added_memory_item_for_log.to_dict_no_embedding()

            memory_item_for_log["memory_action_type"] = "add"
            memory_item_for_log["task_id"] = self.task_state.task_id
            memory_item_for_log["current_stage"] = self.task_state.current_stage
            memory_item_for_log["current_query"] = self.task_state.current_query
            memory_item_for_log["current_turn"] = self.task_state.current_turn
            try:
                append_json_to_jsonl(self.memory_log_dir,memory_item_for_log)
                print("DONE ADD LOG")
            except Exception as e:
                print("append error:", repr(e))

        return ToolResponse(
            content=[TextBlock(type="text", text=note)],
            metadata={"success": True, "memory_id": mem_id, "content": content},
        )

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: dict = None,
    ) -> ToolResponse:
        """Update an existing memory's content and/or metadata in the long term memory store.

        Args:
            memory_id (`str`):
                The target memory identifier.
            content (`str`):
                New content to replace existing content.
            metadata (`dict`, defaults to `None`):
                Metadata to merge/update.

        Returns:
            `ToolResponse`:
                The response containing the result of updating the memory.
        """
        ok = await self.memory_manager.update(memory_id, content, metadata or {})
        if ok:
            note = f"Successfully updated the memory. Memory ID: {memory_id}, updated content: {content}"
            # add update action to memory log.
            if self.memory_log_dir != None and self.memory_log_dir != "":
                memory_item_for_log = {}
                memory_item_for_log["memory_id"] = memory_id
                memory_item_for_log["updated_content"] = content
                memory_item_for_log["memory_action_type"] = "update"
                memory_item_for_log["task_id"] = self.task_state.task_id
                memory_item_for_log["current_stage"] = self.task_state.current_stage
                memory_item_for_log["current_turn"] = self.task_state.current_turn
                append_json_to_jsonl(self.memory_log_dir,memory_item_for_log)
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": True,
                    "memory_id": memory_id,
                    "content": content,
                },
            )
        else:
            note = f"Failed to update the memory. Memory ID: {memory_id} is not found."
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": False,
                    "memory_id": memory_id,
                },
            )

    async def delete_memory(self, memory_id: str, confirmation: bool = False) -> ToolResponse:
        """Delete a memory permanently from the long term memory store  when confirmation is True.

        Args:
            memory_id (`str`):
                The target memory identifier.
            confirmation (`bool`, defaults to `False`):
                Must be True to proceed.

        Returns:
            `ToolResponse`:
                The response containing the result of deleting the memory.
        """
        if not confirmation:
            note = "Deletion is cancelled. The argument 'confirmation' must be True to proceed."
            return ToolResponse(
                content=[TextBlock(type="text", text=note)],
                metadata={"success": False, "memory_id": memory_id},
            )
        ok = await self.memory_manager.delete(memory_id)
        if ok:
            note = f"Successfully deleted the memory. Memory ID: {memory_id}."
            # add delete action to memory log
            if self.memory_log_dir != None and self.memory_log_dir != "":
                memory_item_for_log = {}
                memory_item_for_log["memory_id"] = memory_id
                memory_item_for_log["memory_action_type"] = "delete"
                memory_item_for_log["task_id"] = self.task_state.task_id
                memory_item_for_log["current_stage"] = self.task_state.current_stage
                memory_item_for_log["current_turn"] = self.task_state.current_turn
                append_json_to_jsonl(self.memory_log_dir,memory_item_for_log)
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": True,
                    "memory_id": memory_id,
                },
            )
        else:
            note = f"Failed to delete the memory. Memory ID: {memory_id} is not found."
            return ToolResponse(
                content=[
                    TextBlock(type="text", text=note),
                ],
                metadata={
                    "success": False,
                    "memory_id": memory_id,
                },
            )

    async def _retrieve_from_long_term_memory(self, msg: Msg | list[Msg] | None) -> None:
        """Retrieve relevant memories by semantic similarity from the long term memory store and attach them to context."""
        if msg:
            query = str()
            if isinstance(msg, Msg):
                msg = [msg]
            for m in msg:
                query += m.get_text_content() + "\n"
            items = await self.memory_manager.retrieve(query, 3)
            if items:
                retrieved_block = "\n".join(f"- {it.content} (Memory ID: {it.memory_id})" for it in items)

                memory_content = "<long_term_memory>The content below are retrieved from long-term memory, which maybe " \
                    + f"useful:\n{retrieved_block}</long_term_memory>"

                self._append_context("tool", memory_content, "assistant")

    async def _summarizing(self) -> Msg:
        """Generate a response when the agent fails to solve the problem in
        the maximum iterations."""
        hint_msg = Msg(
            "user",
            "You have failed to generate response within the maximum "
            "iterations. Now respond directly by summarizing the current "
            "situation.",
            role="user",
        )

        # Generate a reply by summarizing the current situation
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *self.context_messages,
                hint_msg,
            ],
        )
       #  finish_function here
        res = await self.model(prompt)

        res_msg = Msg(self.name, [], "assistant")
        if isinstance(res, AsyncGenerator):
            async for chunk in res:
                res_msg.content = chunk.content
        else:
            res_msg.content = res.content
        
        logger.info(f"res_msg: {res_msg}")
        has_answer = TextBlock(type="text", text="has_answer:No")
        res_msg.content.append(has_answer)

        return res_msg

    async def save_memory(self, dir: str = ".\\logs\\memory.jsonl") -> bool:
        """Save the agent's memory state for persistence."""
        memory_list = await self.memory_manager.get_memory()
        array_to_save = []
        for mem in memory_list:
            array_to_save.append({
                "memory_id": mem.memory_id,
                "content": mem.content,
                "metadata": mem.metadata
            })
        if len(array_to_save) == 0:
            return False
        else:
            append_json_array_to_jsonl(array_to_save, dir)
        return True

    def disable_memory_tool(self, tool_name: str) -> None:
        """Disable a memory tool.

        Args:
            tool_name (str): The name of the memory tool to disable. It must be in self.memory_tool_names.

        Raises:
            ValueError: _description_
        """
        if tool_name not in self.memory_tool_names:
            raise ValueError(f"Unknown memory tool: {tool_name}")
        
        if tool_name in self.enabled_memory_tools:
            self.enabled_memory_tools.discard(tool_name)

    def enable_memory_tool(self, tool_name: str) -> None:
        """Enable a memory tool.

        Args:
            tool_name (str): The name of the memory tool to enable. It must be in self.memory_tool_names.

        Raises:
            ValueError: _description_
        """
        if tool_name not in self.memory_tool_names:
            raise ValueError(f"Unknown memory tool: {tool_name}")
        
        if tool_name not in self.enabled_memory_tools:
            self.enabled_memory_tools.add(tool_name)

    def _get_active_tool_schemas(self) -> list[dict]:
        """Dynamically get the enabled memories.

        Returns:
            list[dict]: list of enabled.
        """
        schemas = self.toolkit.get_json_schemas()
        return [
            schema
            for schema in schemas
            if (
                schema.get("function", {}).get("name") not in self.memory_tool_names
                or schema.get("function", {}).get("name") in self.enabled_memory_tools
            )
        ]