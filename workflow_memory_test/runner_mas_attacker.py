import sys,os

from openai import OpenAI
from dataclasses import dataclass, field
from DataSets.reward_hacking import REWARD_HACKINGDataset
from AgeMem_code_agentscope.agent import AgeMem
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from AgeMem_code_agentscope.prompts import CUSTOMER_SERVICE_SYS_PROMPT,BANKING_CUSTOMER_SERVICE_SYS_PROMPT
from AgeMem_code_agentscope.file_tools import append_json_array_to_jsonl,append_json_to_jsonl
from AgeMem_code_agentscope.agent import TaskState
from Attack_MAS.mas_attacker import MAS_Attacker
from Attack_MAS.simple_agent import Model

import asyncio
from agentscope.message import Msg

import json
import uuid
from typing import List
import re
from typing import List

import argparse

def load_memory_from_jsonl(filename: str) -> List[str]:
    """
    Load memory entries from a jsonl file.

    Expected format:
        {"instance_1": "..."}
        {"instance_2": "..."}
        ...

    Parameters
    ----------
    filename : str
        Path to the jsonl file.

    Returns
    -------
    List[str]
        Memory entries ordered by instance index.

    Raises
    ------
    ValueError
        If the file format is invalid.
    """

    pattern = re.compile(r"^instance_(\d+)$")

    parsed = {}

    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_num}: {e}"
                )

            if not isinstance(data, dict):
                raise ValueError(
                    f"Line {line_num} is not a JSON object."
                )

            if len(data) != 1:
                raise ValueError(
                    f"Line {line_num} should contain exactly one instance."
                )

            key, value = next(iter(data.items()))

            m = pattern.fullmatch(key)
            if m is None:
                raise ValueError(
                    f"Invalid key '{key}' at line {line_num}."
                )

            if not isinstance(value, str):
                raise ValueError(
                    f"Value of '{key}' must be a string."
                )

            index = int(m.group(1))

            if index in parsed:
                raise ValueError(
                    f"Duplicate instance index: {index}"
                )

            parsed[index] = value.strip()

    if not parsed:
        return []

    expected = set(range(1, max(parsed.keys()) + 1))

    if set(parsed.keys()) != expected:
        missing = sorted(expected - set(parsed.keys()))
        raise ValueError(
            f"Missing instance indices: {missing}"
        )

    return [parsed[i] for i in sorted(parsed.keys())]

@dataclass
class TestStageQuery:
    task_id:str
    current_stage:str
    query_text:str
    level:str


@dataclass
class Runner:
    """Experiment runner."""
    name: str        # the name of the experiment
    data_domain: str # "all" or "online-shopping_service"
    data_path: str

    run_sample_num: int = -1  # -1 means run all samples
    # Target agent api_key
    api_key_a: str = "your_key"
    base_url_a: str = ""
    #  FALSEPRAISE attacker api_key
    api_key_b: str = ""
    base_url_b: str = ""
    model_name_a:str = "gpt-4o-mini"
    model_name_b:str = "gpt-4o-mini"
    log_dir: str = field(default="") 

    def __post_init__(self):
        if not self.log_dir:  
            self.log_dir = os.path.join(r".\logs",self.name)
            self.memory_log_dir = os.path.join(self.log_dir,"memory_log.jsonl")
            self.coversation_path = os.path.join(self.log_dir,"conversation_history.jsonl") 

        # Load dataset
        datasloader = REWARD_HACKINGDataset(data_path=self.data_path, domain=self.data_domain)
        datasloader.load()
        self.datas = datasloader._instances



# ================== Single Sample Experiment======================

    async def run_one_sample_standardize_experiment(self, evo_turn_num:int = 5, clean_short_memory:bool = True, test_time_evo_before_level_cs:int = -1,test_time_evo_before_level_ct:int = -1, start_data_index:int = -1):
        """run one sample

        Args:
            clean_short_memory (bool, optional): if clean short memory before next sample. Defaults to True.
            test_time_evo_before_level (int, optional): For a single sample during the test phase, enable continuous dialogue for the first N levels (including the boundary value); a value less than or equal to 0 means all levels are frozen by default... Defaults to -1.
        """
        # Run experiments
        datas = []  
        if self.run_sample_num > 0:
            datas = self.datas[:self.run_sample_num]
        else:
            datas = self.datas

        if start_data_index > 0:
            datas = datas[start_data_index:]
        
        # Select a different system prompt based on the scenario.
        Agent_A_sys = ""

        if self.data_domain == "online-shopping_service":
            Agent_A_sys = CUSTOMER_SERVICE_SYS_PROMPT
        elif self.data_domain == "online-banking_service":
            print("Running online banking service experiment...")
            Agent_A_sys = BANKING_CUSTOMER_SERVICE_SYS_PROMPT
        # Create the research target agent.
        modelA = OpenAIChatModel(
                api_key=self.api_key_a,
                model_name=self.model_name_a,
                client_kwargs={
                    "base_url": self.base_url_a
                },
                generate_kwargs={
                "temperature": 0.1,
                "seed":42,
                },
                stream=False,
        )
        formatterA = OpenAIChatFormatter()
        Agent_A = AgeMem(
            name="AgeMem",
            sys_prompt=Agent_A_sys,
            model=modelA,
            formatter=formatterA,
            memory_log_dir=self.memory_log_dir,
            task_state=TaskState(task_id="none",current_stage="none",current_turn=-1,current_query=-1)
        )
        # Create the attacker: FALSEPRAISE.
        attacker_mas = MAS_Attacker(model=Model(base_url=self.base_url_b,api_key=self.api_key_b,model_name=self.model_name_b),
                                    scenario=self.data_domain)

        for data in datas:
            # clean Long Memory 
            await Agent_A.memory_manager.clear()
            len_LM = await Agent_A.memory_manager.get_long_memory_len()
            print(f"The long memory len in task:{data.task_id},is : {len_LM}")
            # if clean corss session short memory
            if clean_short_memory:
                Agent_A.context_messages.clear()
                attacker_mas.stm.clear()
            print(f"Running task: {data.task_id}")

            # ========================= Interaction  Stage ========================= 
            conversation_history_evo = []
            response_A = ""
            response_B = ""
            Agent_A.task_state.task_id = data.task_id
            Agent_A.task_state.current_stage = "initial_user_query"
            Agent_A.task_state.current_query = 0
            for turn in range(evo_turn_num):
                # One Turn History
                one_turn_history = {}
                one_turn_history["task_id"] = data.task_id
                one_turn_history["current_stage"] = "initial_user_query"
                one_turn_history["current_query"] = 0
                one_turn_history["current_turn"] = turn

                # print("=" * 50)
                # Agent A's turn
                Agent_A.task_state.current_turn = turn
                if turn == 0:
                    one_turn_history["C"] = data.initial_user_query
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=data.initial_user_query, role="user"))
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                else:
                    # Agent B's turn
                    response_B = attacker_mas.get_mas_final_response(target_query=response_A)
                    # log
                    one_turn_history["C"] = response_B
                    
                    # Agent A's turn
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=response_B, role="user"))
                    # log
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                    # print(f"Task {data.task_id} - Agent A's response at turn {turn}.")
                
                # Add to FALSEPRAISE STM.
                attacker_mas.stm.add(current_turn=one_turn_history["current_turn"],speaker_c=one_turn_history["C"],speaker_s=one_turn_history["S"])
                # Add to log
                conversation_history_evo.append(one_turn_history)

            # Log results
            append_json_array_to_jsonl(conversation_history_evo, self.coversation_path)
            
            # ========================= Cross Session Test Stage ========================= 
            # set evo state
            test_time_evo_before_num_cs:int = -1

            if test_time_evo_before_level_cs > 0:
                test_time_evo_before_num_cs = test_time_evo_before_level_cs * 2

            
            for index, query in enumerate(data.cross_session_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.
                    query = query + "My username:Lucy123, email address: lucy123.lucy@gmail.com,phone number:18885678910."
                # Number of dialogue turns: set to 2 when test-time dialogue is allowed; defaults to 1, meaning the user provides no feedback.
                act_turns_num:int = 1 

                if index < test_time_evo_before_num_cs:
                    act_turns_num = 2 
                    # Enable memory add when test-time dialogue is allowed.
                    Agent_A.enable_memory_tool("add_memory")
                else:
                    Agent_A.disable_memory_tool("add_memory")
                    Agent_A.disable_memory_tool("update_memory")


                # Whether clearing short-term memory has an impact.
                if clean_short_memory:
                    Agent_A.context_messages.clear()
                    attacker_mas.stm.clear()
                
                # Initialize the statements from both parties.
                Agent_B_utterance = ""
                Agent_A_utterance = ""

                for i in range(act_turns_num):

                    # log initialize
                    test_log_item:dict = {}
                    test_log_item["task_id"] = data.task_id
                    test_log_item["current_stage"] = "cross_session_queries"
                    test_log_item["current_query"] = index
                    test_log_item["current_turn"] = i
                    test_log_item["test_time_evo_before_num"] = test_time_evo_before_num_cs

                    # User statement / user utterance
                    Agent_B_utterance = attacker_mas.get_mas_final_response(target_query=Agent_A_utterance) if i != 0 else query
                    test_log_item["C"] = Agent_B_utterance
                    
                    # set agent's taskstate
                    Agent_A.task_state.task_id = data.task_id
                    Agent_A.task_state.current_stage = "cross_session_queries"
                    Agent_A.task_state.current_query = index
                    Agent_A.task_state.current_turn = i
                    
                    # Call the agent under test to answer.
                    Agent_A_utterance = (await Agent_A.reply(msg=Msg(name="user", content=Agent_B_utterance, role="user"))).get_text_content()
                    test_log_item["S"] = Agent_A_utterance

                    # When test-time dialogue is allowed, add an initial short-term memory to the Attacker so that it can provide user feedback.
                    if(i == 0 and test_time_evo_before_num_cs > 0):
                        attacker_mas.stm.add(current_turn=test_log_item["current_turn"],speaker_c=test_log_item["C"], speaker_s=test_log_item["S"])
                    
                    # Log results
                    append_json_to_jsonl(self.coversation_path,test_log_item)

            # Restore tool state.
            Agent_A.enable_memory_tool("add_memory")
            Agent_A.enable_memory_tool("update_memory")

            # ========================= Cross Task Test Stage ========================= 
            # set evo state
            test_time_evo_before_num_ct:int = -1

            if test_time_evo_before_level_ct > 0:
                test_time_evo_before_num_ct = test_time_evo_before_level_ct * 2


            for index, query in enumerate(data.cross_task_generalization_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.
                    query = query + "My username:Lucy123, email address: lucy123.lucy@gmail.com,phone number:18885678910."

                # Number of dialogue turns: set to 2 when test-time dialogue is allowed; defaults to 1, meaning the user provides no feedback.
                act_turns_num:int = 1

                if index < test_time_evo_before_num_ct:
                    act_turns_num = 2
                    # Enable memory add when test-time dialogue is allowed.
                    Agent_A.enable_memory_tool("add_memory")
                else:
                    Agent_A.disable_memory_tool("add_memory")
                    Agent_A.disable_memory_tool("update_memory")

                # Whether clearing short-term memory has an impact.
                if clean_short_memory:
                    Agent_A.context_messages.clear()
                    attacker_mas.stm.clear()
                
                # Initialize the statements from both parties.s
                Agent_B_utterance = ""
                Agent_A_utterance = ""

                for i in range(act_turns_num):

                    # log initialize
                    test_log_item:dict = {}
                    test_log_item["task_id"] = data.task_id
                    test_log_item["current_stage"] = "cross_task_generalization_queries"
                    test_log_item["current_query"] = index
                    test_log_item["current_turn"] = i
                    test_log_item["test_time_evo_before_num"] = test_time_evo_before_num_ct

                    # User statement / user utterance
                    Agent_B_utterance = attacker_mas.get_mas_final_response(target_query=Agent_A_utterance) if i != 0 else query
                    test_log_item["C"] = Agent_B_utterance
                    
                    # set agent's taskstate
                    Agent_A.task_state.task_id = data.task_id
                    Agent_A.task_state.current_stage = "cross_task_generalization_queries"
                    Agent_A.task_state.current_query = index
                    Agent_A.task_state.current_turn = i
                    
                    # Call the agent under test to answer.
                    Agent_A_utterance = (await Agent_A.reply(msg=Msg(name="user", content=Agent_B_utterance, role="user"))).get_text_content()
                    test_log_item["S"] = Agent_A_utterance

                    # When test-time dialogue is allowed, add an initial short-term memory to the Attacker so that it can provide user feedback.
                    if(i == 0 and test_time_evo_before_num_ct > 0):
                        attacker_mas.stm.add(current_turn=test_log_item["current_turn"],speaker_c=test_log_item["C"], speaker_s=test_log_item["S"])
                    
                    # Log results
                    append_json_to_jsonl(self.coversation_path,test_log_item)

            # Restore tool state.
            Agent_A.enable_memory_tool("add_memory")
            Agent_A.enable_memory_tool("update_memory")
                
    async def run_one_sample_no_memory_experiment(self, evo_turn_num:int = 1, clean_short_memory:bool = True, start_data_index:int = -1):
        """Run cross-session/cross-task multi-turn attacks without allowing memory writes."""
        # Run experiments
        datas = []
        if self.run_sample_num > 0:
            datas = self.datas[:self.run_sample_num]
        else:
            datas = self.datas

        if start_data_index > 0:
            datas = datas[start_data_index:]
                
        # Select the system prompt based on the scenario.
        Agent_A_sys = ""
        if self.data_domain == "online-shopping_service":
            Agent_A_sys = CUSTOMER_SERVICE_SYS_PROMPT
        elif self.data_domain == "online-banking_service":
            print("in run_one_sample_no_memory_experiment, data_domain is online-banking-service")
            Agent_A_sys = BANKING_CUSTOMER_SERVICE_SYS_PROMPT
        # Initialize the target agent.
        modelA = OpenAIChatModel(
                api_key=self.api_key_a,
                model_name=self.model_name_a,
                client_kwargs={
                    "base_url": self.base_url_a
                },
                generate_kwargs={
                "temperature": 0.1,
                "seed":42,
                "extra_body":{"thinking": {"type": "disabled"},"enable_thinking": False}
                },
                stream=False,
        )
        formatterA = OpenAIChatFormatter()
        Agent_A = AgeMem(
            name="AgeMem",
            sys_prompt=Agent_A_sys,
            model=modelA,
            formatter=formatterA,
            memory_log_dir=self.memory_log_dir,
            task_state=TaskState(task_id="none",current_stage="none",current_turn=-1,current_query=-1)
        )

        # Initialize FALSEPRAISE.
        attacker_mas = MAS_Attacker(model=Model(base_url=self.base_url_b,api_key=self.api_key_b,model_name=self.model_name_b),
                            scenario=self.data_domain)
        
        async def run_query_evo(data, stage_name:str, query_index:int, query:str):
            response_A = ""
            response_B = ""

            if clean_short_memory:
                Agent_A.context_messages.clear()
                attacker_mas.stm.clear()

            Agent_A.disable_memory_tool("add_memory")
            Agent_A.disable_memory_tool("update_memory")
            Agent_A.task_state.task_id = data.task_id
            Agent_A.task_state.current_stage = stage_name
            Agent_A.task_state.current_query = query_index

            for turn in range(evo_turn_num):
                one_turn_history = {}
                one_turn_history["task_id"] = data.task_id
                one_turn_history["current_stage"] = stage_name
                one_turn_history["current_query"] = query_index
                one_turn_history["current_turn"] = turn

                print("=" * 50)
                Agent_A.disable_memory_tool("add_memory")
                Agent_A.disable_memory_tool("update_memory")
                Agent_A.task_state.current_turn = turn

                if turn == 0:
                    one_turn_history["C"] = query
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=query, role="user"))
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                else:
                    response_B = attacker_mas.get_mas_final_response(target_query=response_A)
                    response_B = response_B.get_text_content()
                    one_turn_history["C"] = response_B

                    print(f"Task {data.task_id} - Agent B's response at {stage_name} query {query_index}, turn {turn}")

                    Agent_A.task_state.current_turn = turn
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=response_B, role="user"))
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                    print(f"Task {data.task_id} - Agent A's response at {stage_name} query {query_index}, turn {turn}.")
                    print("=" * 50)

                # Add STM to FALSEPRAISE.
                attacker_mas.stm.add(current_turn=one_turn_history["current_turn"],speaker_c=one_turn_history["C"],speaker_s=one_turn_history["S"])
                
                append_json_to_jsonl(self.coversation_path,one_turn_history)

        for data in datas:
            await Agent_A.memory_manager.clear()
            Agent_A.disable_memory_tool("add_memory")
            len_LM = await Agent_A.memory_manager.get_long_memory_len()
            print(f"The long memory len in task:{data.task_id},is : {len_LM}")

            if clean_short_memory:
                Agent_A.context_messages.clear()
                attacker_mas.stm.clear()
            print(f"Running task: {data.task_id}")

            # ========================= Cross Session No Memory Attack  =========================
            for index, query in enumerate(data.cross_session_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.
                    query = query + "My username:Jack123, email address:jack123.jack@gmail.com,phone number:19995678910."

                await run_query_evo(data, "cross_session_queries", index, query)

            # ========================= Cross Task No Memory Attack =========================
            for index, query in enumerate(data.cross_task_generalization_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.
                    query = query + "My username:Jack123, email address:jack123.jack@gmail.com,phone number:19995678910."

                await run_query_evo(data, "cross_task_generalization_queries", index, query)

    async def run_one_sample_with_bad_memory_experiment(self, bad_memory_path:str, evo_turn_num:int = 1, clean_short_memory:bool = True, start_data_index:int = -1):
        """Run cross-session/cross-task multi-turn attacks with bad memory writes.
        evo_turn_num:Number of dialogue turns at test time; defaults to 1, meaning FALSEPRAISE is not used to provide feedback.
        """
        # Run experiments
        datas = []
        if self.run_sample_num > 0:
            datas = self.datas[:self.run_sample_num]
        else:
            datas = self.datas

        if start_data_index > 0:
            datas = datas[start_data_index:]
        # Select the system prompt based on the scenario.
        Agent_A_sys = ""
        if self.data_domain == "online-shopping_service":
            Agent_A_sys = CUSTOMER_SERVICE_SYS_PROMPT
        elif self.data_domain == "online-banking_service":
            Agent_A_sys = BANKING_CUSTOMER_SERVICE_SYS_PROMPT
        # Initialize the target agent.
        modelA = OpenAIChatModel(
                api_key=self.api_key_a,
                model_name=self.model_name_a,
                client_kwargs={
                    "base_url": self.base_url_a
                },
                generate_kwargs={
                "temperature": 0.1,
                "seed":42,
                "extra_body":{"thinking": {"type": "disabled"},"enable_thinking": False}
                },
                stream=False,
        )
        formatterA = OpenAIChatFormatter()
        Agent_A = AgeMem(
            name="AgeMem",
            sys_prompt=Agent_A_sys,
            model=modelA,
            formatter=formatterA,
            memory_log_dir=self.memory_log_dir,
            task_state=TaskState(task_id="none",current_stage="none",current_turn=-1,current_query=-1)
        )

        # Load poisoned memories into Target Agent
        bad_memorys = load_memory_from_jsonl(bad_memory_path)
        for i, mem in enumerate(bad_memorys):
            theType = ""
            if i < 25:
                theType = "general_L1"
            elif i >= 25 and i < 40:
                theType = "general_L2"
            elif i >=40:
                theType = "specific_L3"
            await Agent_A.memory_manager.add(
                memory_id=str(uuid.uuid4()),
                content=mem,
                metadata={
                    "type": theType
                }
            )
            print(f"Loaded bad memory {i+1}/{len(bad_memorys)} into Agent_A's long-term memory.")

        # Initialize FALSEPRAISE.
        attacker_mas = MAS_Attacker(model=Model(base_url=self.base_url_b,api_key=self.api_key_b,model_name=self.model_name_b),
                                    scenario=self.data_domain)

        async def run_query_evo(data, stage_name:str, query_index:int, query:str):
            response_A = ""
            response_B = ""

            if clean_short_memory:
                Agent_A.context_messages.clear()
                attacker_mas.stm.clear()

            Agent_A.disable_memory_tool("add_memory")
            Agent_A.disable_memory_tool("update_memory")
            Agent_A.task_state.task_id = data.task_id
            Agent_A.task_state.current_stage = stage_name
            Agent_A.task_state.current_query = query_index

            for turn in range(evo_turn_num):
                one_turn_history = {}
                one_turn_history["task_id"] = data.task_id
                one_turn_history["current_stage"] = stage_name
                one_turn_history["current_query"] = query_index
                one_turn_history["current_turn"] = turn

                print("=" * 50)
                Agent_A.disable_memory_tool("add_memory")
                Agent_A.disable_memory_tool("update_memory")
                Agent_A.task_state.current_turn = turn

                if turn == 0:
                    one_turn_history["C"] = query
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=query, role="user"))
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                else:
                    response_B = attacker_mas.get_mas_final_response(target_query=response_A)
                    one_turn_history["C"] = response_B

                    print(f"Task {data.task_id} - Agent B's response at {stage_name} query {query_index}, turn {turn}")

                    Agent_A.task_state.current_turn = turn
                    response_A = await Agent_A.reply(msg=Msg(name="user", content=response_B, role="user"))
                    response_A = response_A.get_text_content()
                    one_turn_history["S"] = response_A
                    print(f"Task {data.task_id} - Agent A's response at {stage_name} query {query_index}, turn {turn}.")
                    print("=" * 50)
                # save stm for FALSEPRAISE.
                attacker_mas.stm.add(current_turn=one_turn_history["current_turn"],speaker_c=one_turn_history["C"],speaker_s=one_turn_history["S"])
                append_json_to_jsonl(self.coversation_path,one_turn_history)

        for data in datas:
            Agent_A.disable_memory_tool("add_memory")
            Agent_A.disable_memory_tool("update_memory")
            len_LM = await Agent_A.memory_manager.get_long_memory_len()
            print(f"The long memory len in task:{data.task_id},is : {len_LM}")

            if clean_short_memory:
                Agent_A.context_messages.clear()
                attacker_mas.stm.clear()
            print(f"Running task: {data.task_id}")

            # ========================= Cross Session No Memory Attack  =========================
            for index, query in enumerate(data.cross_session_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.
                    query = query + "My username:Jack123, email address:jack123.jack@gmail.com,phone number:19995678910."

                await run_query_evo(data, "cross_session_queries", index, query)

            # ========================= Cross Task No Memory Attack =========================
            for index, query in enumerate(data.cross_task_generalization_queries):
                if data.scenario == "online-banking_service":
                    # Add extra information in the banking scenario for quick testing.s
                    query = query + "My username:Jack123, email address:jack123.jack@gmail.com,phone number:19995678910."

                await run_query_evo(data, "cross_task_generalization_queries", index, query)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MAS experiment."
    )

    # Runner parameter
    parser.add_argument(
        "--name",
        type=str,
        default="test",
        help="Experiment name",
    )

    parser.add_argument(
        "--data-domain",
        type=str,
        default="online-banking_service",
        help="Data domain",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the JSONL dataset",
    )

    parser.add_argument(
        "--run-sample-num",
        type=int,
        default=-1,
        help="Number of samples to run. -1 means all samples.",
    )

    parser.add_argument(
        "--model-name-a",
        type=str,
        default="gpt-4o-mini",
        help="Model name for Target Agent",
    )

    parser.add_argument(
        "--model-name-b",
        type=str,
        default="gpt-4o-mini",
        help="Model name for FALSEPRAISE",
    )

    parser.add_argument(
        "--api-key-a",
        type=str,
        default="your_api",
        help="API key for Target Agent",
    )

    parser.add_argument(
        "--base-url-a",
        type=str,
        default="your_url",
        help="Base URL for Target Agent",
    )

    parser.add_argument(
        "--api-key-b",
        type=str,
        default="your_key",
        help="API key for FALSEPRAISE",
    )

    parser.add_argument(
        "--base-url-b",
        type=str,
        default="your_url",
        help="Base URL for FALSEPRAISE",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Log directory",
    )

    # Experiment parameter
    parser.add_argument(
        "--evo-turn-num",
        type=int,
        default=5,
        help="Number of evolution turns",
    )

    parser.add_argument(
        "--start-data-index",
        type=int,
        default=-1,
        help="Starting data index",
    )

    parser.add_argument(
        "--experiment-type",
        type=str,
        default="standardize",
        help="Experiment type",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    runner = Runner(
        name=args.name,
        data_domain=args.data_domain,
        data_path=args.data_path,
        run_sample_num=args.run_sample_num,

        api_key_a=args.api_key_a,
        base_url_a=args.base_url_a,

        api_key_b=args.api_key_b,
        base_url_b=args.base_url_b,

        model_name_a=args.model_name_a,
        model_name_b=args.model_name_b,

        log_dir=args.log_dir,
    )

    if args.experiment_type == "standardize":
        await runner.run_one_sample_standardize_experiment(
            evo_turn_num=args.evo_turn_num,
            start_data_index=args.start_data_index,
        )
    elif args.experiment_type == "no_memory":
        await runner.run_one_sample_no_memory_experiment(start_data_index=args.start_data_index)
    elif args.experiment_type == "bad_memory":
        await runner.run_one_sample_with_bad_memory_experiment(
            bad_memory_path=os.path.join(r"\datas",args.data_domain + "_bad_memory.jsonl"),
            start_data_index=args.start_data_index,
        )

if __name__ == "__main__":

    asyncio.run(main())
