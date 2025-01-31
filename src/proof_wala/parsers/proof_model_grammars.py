import typing
import copy
from enum import Enum
from itp_interface.tools.training_data_format import TrainingDataFormat, Goal
try:
    from .grammars.grammar import Grammar
except ImportError:
    from grammars.grammar import Grammar


class CustomGoalHash(object):
    def __init__(self, goal: Goal):
        self.goal = goal

    def __hash__(self):
        return hash(self.goal.goal)

    def __eq__(self, other):
        if not isinstance(other, CustomGoalHash):
            return False
        return self.goal == other.goal

def get_delta_state(start_goals: typing.List[Goal], end_goals: typing.List[Goal]) -> typing.List[Goal]:
    delta_goals = []
    start_goals : typing.List[CustomGoalHash] = [CustomGoalHash(goal) for goal in start_goals]
    end_goals : typing.List[CustomGoalHash] = [CustomGoalHash(goal) for goal in end_goals]
    start_goal_set = set(start_goals)
    end_goal_set = set(end_goals)
    new_goals = end_goal_set.difference(start_goal_set)
    new_goal_ordered = [(idx, True, goal.goal) for idx, goal in enumerate(end_goals) if goal in new_goals]
    same_goals = start_goal_set.intersection(end_goal_set)
    same_goal_ordered = [(idx, False, goal.goal) for idx, goal in enumerate(end_goals) if goal in same_goals]
    all_goals_ordered = same_goal_ordered + new_goal_ordered
    # Sort the goals based on the order of the goals in the end_goals
    all_goals_ordered = sorted(all_goals_ordered, key=lambda x: x[0])
    delta_goals = []
    for _, is_new, goal in all_goals_ordered:
        if is_new:
            delta_goals.append(goal)
        else:
            goal_idx = start_goals.index(CustomGoalHash(goal))
            assert goal_idx >= 0 and goal_idx < len(start_goals), f"Goal index {goal_idx} is out of bounds"
            delta_goals.append(Goal(goal=f"DITTO {goal_idx}", hypotheses=[]))
    return delta_goals

def get_full_state(start_goals: typing.List[Goal], delta_goals: typing.List[Goal]) -> typing.List[Goal]:
    full_goals = []
    for goal in delta_goals:
        if goal.goal.startswith("DITTO"):
            idx = int(goal.goal.split(" ")[-1])
            full_goals.append(start_goals[idx])
        else:
            full_goals.append(goal)
    return full_goals


class ProofModelGrammar(Grammar):
    grammar = """
Prog:
   StatesStr ProofState End
|  StatesStr ProofState ProofStep End;
ProofState:
  Description String States
| States;
States:
  State
| State States
| EMPTY;
State:
 StateNum String HypsResponses;
HypsResponses:
    Hyps HypResponses
|   EMPTY;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;
ProofStep: 
ProofStepStr String;
StateNum:
    StateStr int;


terminals
int: /\d+/;
StateStr: "[STATE]";
StatesStr: "[STATES]";
ProofStepStr: "[PROOFSTEP]";
Hyps: "[HYPOTHESES]";
Hyp: "[HYPOTHESIS]";
End: "[END]";
Description: "[DESCRIPTION]";
String:;
"""
    class Keywords(Enum):
        STATE = "[STATE]"
        STATES = "[STATES]"
        PROOFSTEP = "[PROOFSTEP]"
        HYPOTHESES = "[HYPOTHESES]"
        HYPOTHESIS = "[HYPOTHESIS]"
        SUCCESS = "[SUCCESS]"
        END = "[END]"
        DESCRIPTION = "[DESCRIPTION]"

        def __str__(self) -> str:
            return self.value

    keywords = [keyword.value for keyword in Keywords]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in ProofModelGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1

    def __init__(self, state_delta: bool = False, enable_proofstep: bool = True, ordering: typing.List[str] = ["DESCRIPTION", "STATE", "PROOFSTEP"]):
        recognizers = {
            'String': ProofModelGrammar.before_keyword
        }
        self.state_delta = state_delta
        self.enable_proofstep = enable_proofstep
        self.ordering = ordering
        super(ProofModelGrammar, self).__init__(ProofModelGrammar.grammar, ProofModelGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, training_data_format: TrainingDataFormat, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            ProofModelGrammar.Keywords.DESCRIPTION : [],
            ProofModelGrammar.Keywords.STATE : [],
            ProofModelGrammar.Keywords.PROOFSTEP : [],
        }
        lines_order = []
        try:
            for keyword in self.ordering:
                keyword_enum = ProofModelGrammar.Keywords[str(keyword)]
                if keyword_enum not in lines_order:
                    lines_order.append(keyword_enum)
        except:
            lines_order = [
                ProofModelGrammar.Keywords.DESCRIPTION,
                ProofModelGrammar.Keywords.STATE,
                ProofModelGrammar.Keywords.PROOFSTEP,
            ]
        priority_order_lo_hi = [
            ProofModelGrammar.Keywords.STATE,
            ProofModelGrammar.Keywords.PROOFSTEP,
        ]
        lines_map[ProofModelGrammar.Keywords.STATE] = [str(ProofModelGrammar.Keywords.STATES)]
        if training_data_format.goal_description is not None:
            new_line = f"{ProofModelGrammar.Keywords.DESCRIPTION}\n{training_data_format.goal_description}\n"
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
        for i, goal in enumerate(training_data_format.start_goals):
            new_line = f"{ProofModelGrammar.Keywords.STATE} {i}" # Add the goal number
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            new_line = str(goal.goal)
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            if len(goal.hypotheses) > 0:
                new_line = f"{ProofModelGrammar.Keywords.HYPOTHESES}"
                lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
                for hyp in goal.hypotheses:
                    new_line = f"{ProofModelGrammar.Keywords.HYPOTHESIS} {hyp}"
                    lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
        if len(training_data_format.proof_steps) > 0 and self.enable_proofstep:
            new_line = f"\n{ProofModelGrammar.Keywords.PROOFSTEP}"
            lines_map[ProofModelGrammar.Keywords.PROOFSTEP] = [new_line]
            lines_map[ProofModelGrammar.Keywords.PROOFSTEP].extend(training_data_format.proof_steps)
        keywords = [keyword for keyword in lines_map.keys()]
        # Convert all the lines under each keyword to a single string
        for keyword in keywords:
            lines_map[keyword] = "\n".join(lines_map[keyword])
        # Frame the first prompt version without any token limit
        text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{ProofModelGrammar.Keywords.END}"
        
        # Trim the lines based on the max_token_cnt
        if char_cnt is not None and len(text) > char_cnt:
            _idx = 0
            diff = len(text) - char_cnt
            while _idx < len(priority_order_lo_hi) and diff > 0:
                trim_part = priority_order_lo_hi[_idx]
                if trim_part in lines_map:
                    lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{ProofModelGrammar.Keywords.END}"
                diff = len(text) - char_cnt
                _idx += 1

        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None:
            assert len(text) <= char_cnt, f"Text length {len(text)} is greater than the max token count {char_cnt}. Possibly too few characters per token." +\
            f" characters_per_token = {characters_per_token}, max_token_cnt = {max_token_cnt}"
            # text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text

    def _parse_expr(self, rule_type, nodes, context: TrainingDataFormat):
        if rule_type == ProofModelGrammar.Keywords.HYPOTHESIS:
            hyps = []
            for node in nodes:
                hyps.extend(node)
            return hyps
        elif rule_type == ProofModelGrammar.Keywords.HYPOTHESES:
            return nodes[1] if len(nodes) > 1 else []
        elif rule_type == ProofModelGrammar.Keywords.STATE:
            hyps = nodes[2] if len(nodes) > 2 else []
            goal = nodes[1].strip() if len(nodes) > 1 else None
            state = Goal(hypotheses=hyps, goal=goal)
            return [state]
        elif rule_type == ProofModelGrammar.Keywords.STATES:
            states = []
            for node in nodes:
                states.extend(node)
            return states
        elif rule_type == "Prog":
            if len(nodes) >= 1 and nodes[0] == ProofModelGrammar.Keywords.DESCRIPTION.value:
                context.goal_description = nodes[1].strip() if nodes[1] is not None else None
                context.start_goals = nodes[2]
            else:
                context.start_goals = nodes
            return context
        else:
            raise Exception("Not implemented")

    def get_action(self, tdf):
        context = tdf
        actions = {
            "Prog": lambda _, nodes: self._parse_expr("Prog", nodes[1], context),
            "States": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.STATES, nodes, context),
            "State": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.STATE, nodes, context),
            "HypsResponses": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.HYPOTHESES, nodes, context),
            "HypResponses": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.HYPOTHESIS, nodes, context),
            "HypResponse": lambda _, nodes: [nodes[1]],
            "String": lambda _, nodes: str(nodes).strip(),
        }
        return actions

    def interpret_result(self, result):
        return result
    
    def parse(self, text, tdf: TrainingDataFormat = None) -> TrainingDataFormat:
        tdf = TrainingDataFormat() if tdf is None else copy.deepcopy(tdf)
        final_tdf : TrainingDataFormat = self.run(text, tdf)
        return final_tdf


class ProofModelPredGrammar(Grammar):
    grammar = """
Prog:
   ProofStep StatesStr ProofState End
|  StatesStr ProofState End;
ProofState:
  Description String States
| States;
States:
  State
| State States
| EMPTY;
State:
 StateNum String HypsResponses;
HypsResponses:
    Hyps HypResponses
|   EMPTY;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;
StateNum:
    StateStr int;
ProofStep:
    ProofStepStr String;

terminals
int: /\d+/;
StateStr: "[STATE]";
StatesStr: "[STATES]";
Hyps: "[HYPOTHESES]";
Hyp: "[HYPOTHESIS]";
End: "[END]";
Description: "[DESCRIPTION]";
ProofStepStr: "[PROOFSTEP]";
String:;
"""

    def __init__(self, state_delta: bool = False, enable_proofstep: bool = False, ordering: typing.List[str] = ["PROOFSTEP", "DESCRIPTION", "STATE"]):
        recognizers = {
            'String': ProofModelGrammar.before_keyword
        }
        self.state_delta = state_delta
        self.enable_proofstep = enable_proofstep
        self.ordering = ordering
        super(ProofModelPredGrammar, self).__init__(ProofModelPredGrammar.grammar, ProofModelGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, training_data_format: TrainingDataFormat, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        if self.enable_proofstep:
            lines_map = {
                ProofModelGrammar.Keywords.PROOFSTEP : [],
                ProofModelGrammar.Keywords.DESCRIPTION : [],
                ProofModelGrammar.Keywords.STATE : []
            }
            lines_order = []
            try:
                for keyword in self.ordering:
                    keyword_enum = ProofModelGrammar.Keywords[str(keyword)]
                    if keyword_enum not in lines_order:
                        lines_order.append(keyword_enum)
            except:
                lines_order = [
                    ProofModelGrammar.Keywords.PROOFSTEP,
                    ProofModelGrammar.Keywords.DESCRIPTION,
                    ProofModelGrammar.Keywords.STATE
                ]
            priority_order_lo_hi = [
                ProofModelGrammar.Keywords.PROOFSTEP,
                ProofModelGrammar.Keywords.DESCRIPTION,
                ProofModelGrammar.Keywords.STATE,
            ]
        else:
            lines_map = {
                ProofModelGrammar.Keywords.DESCRIPTION : [],
                ProofModelGrammar.Keywords.STATE : []
            }
            lines_order = []
            try:
                for keyword in self.ordering:
                    keyword_enum = ProofModelGrammar.Keywords[str(keyword)]
                    if keyword_enum not in lines_map:
                        lines_order.append(keyword_enum)
            except:
                lines_order = [
                    ProofModelGrammar.Keywords.DESCRIPTION,
                    ProofModelGrammar.Keywords.STATE
                ]
            priority_order_lo_hi = [
                ProofModelGrammar.Keywords.DESCRIPTION,
                ProofModelGrammar.Keywords.STATE,
            ]
        lines_map[ProofModelGrammar.Keywords.STATE] = [str(ProofModelGrammar.Keywords.STATES)]
        if len(training_data_format.proof_steps) > 0 and self.enable_proofstep:
            new_line = f"{ProofModelGrammar.Keywords.PROOFSTEP}"
            lines_map[ProofModelGrammar.Keywords.PROOFSTEP].append(new_line)
            lines_map[ProofModelGrammar.Keywords.PROOFSTEP].extend(training_data_format.proof_steps)
        additional_info = ""
        if training_data_format.addition_state_info is not None:
            if len(training_data_format.addition_state_info) > 0:
                progress = training_data_format.addition_state_info.get("progress", None)
                if progress is not None:
                    additional_info += f"Progress: {progress}\n"
                error_message = training_data_format.addition_state_info.get("error_message", None)
                if error_message is not None:
                    additional_info += f"Error: {error_message}\n"
        if training_data_format.goal_description is not None or len(additional_info) > 0:
            if training_data_format.goal_description is not None:
                new_line = f"{ProofModelGrammar.Keywords.DESCRIPTION}\n{training_data_format.goal_description}\n"
            else:
                new_line = f"{ProofModelGrammar.Keywords.DESCRIPTION}\n"
            new_line += additional_info
            lines_map[ProofModelGrammar.Keywords.DESCRIPTION].append(new_line)
        end_goals = training_data_format.end_goals if not self.state_delta else get_delta_state(training_data_format.start_goals, training_data_format.end_goals)
        for i, goal in enumerate(end_goals):
            new_line = f"{ProofModelGrammar.Keywords.STATE} {i}"
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            new_line = str(goal.goal)
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            if len(goal.hypotheses) > 0:
                new_line = f"{ProofModelGrammar.Keywords.HYPOTHESES}"
                lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
                for hyp in goal.hypotheses:
                    new_line = f"{ProofModelGrammar.Keywords.HYPOTHESIS} {hyp}"
                    lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
        keywords = [keyword for keyword in lines_map.keys()]
        # Convert all the lines under each keyword to a single string
        for keyword in keywords:
            lines_map[keyword] = "\n".join(lines_map[keyword])
        # Frame the first prompt version without any token limit
        text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{ProofModelGrammar.Keywords.END}"
        
        # Trim the lines based on the max_token_cnt
        if char_cnt is not None and len(text) > char_cnt:
            _idx = 0
            diff = len(text) - char_cnt
            while _idx < len(priority_order_lo_hi) and diff > 0:
                trim_part = priority_order_lo_hi[_idx]
                if trim_part in lines_map:
                    lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{ProofModelGrammar.Keywords.END}"
                diff = len(text) - char_cnt
                _idx += 1

        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None:
            assert len(text) <= char_cnt, f"Text length {len(text)} is greater than the max token count {char_cnt}. Possibly too few characters per token." +\
            f" characters_per_token = {characters_per_token}, max_token_cnt = {max_token_cnt}"
            # text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text

    def _parse_states_node(self, states_node, context: TrainingDataFormat):
        if len(states_node) >= 1 and states_node[0] == ProofModelGrammar.Keywords.DESCRIPTION.value:
            context.goal_description = states_node[1].strip() if states_node[1] is not None else None
            context.end_goals = states_node[2]
        else:
            context.end_goals = states_node

    def _parse_expr(self, rule_type, nodes, context: TrainingDataFormat):
        if rule_type == ProofModelGrammar.Keywords.HYPOTHESIS:
            hyps = []
            for node in nodes:
                hyps.extend(node)
            return hyps
        elif rule_type == ProofModelGrammar.Keywords.HYPOTHESES:
            return nodes[1] if len(nodes) > 1 else []
        elif rule_type == ProofModelGrammar.Keywords.STATE:
            hyps = nodes[2] if len(nodes) > 2 else []
            goal = nodes[1].strip() if len(nodes) > 1 else None
            state = Goal(hypotheses=hyps, goal=goal)
            return [state]
        elif rule_type == ProofModelGrammar.Keywords.STATES:
            states = []
            for node in nodes:
                states.extend(node)
            return states
        elif rule_type == ProofModelGrammar.Keywords.PROOFSTEP:
            context.proof_steps = [nodes[1]]
            return None
        elif rule_type == "Prog":
            node_idx = 0
            while node_idx < len(nodes):
                if nodes[node_idx] == ProofModelGrammar.Keywords.STATES.value:
                    states_node = nodes[node_idx + 1]
                    self._parse_states_node(states_node, context)
                    break
                node_idx += 1
            if node_idx == len(nodes):
                context.end_goals = nodes
            return context
        else:
            raise Exception("Not implemented")

    def get_action(self, tdf):
        context = tdf
        actions = {
            "Prog": lambda _, nodes: self._parse_expr("Prog", nodes, context),
            "States": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.STATES, nodes, context),
            "State": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.STATE, nodes, context),
            "ProofStep": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.PROOFSTEP, nodes, context),
            "HypsResponses": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.HYPOTHESES, nodes, context),
            "HypResponses": lambda _, nodes: self._parse_expr(ProofModelGrammar.Keywords.HYPOTHESIS, nodes, context),
            "HypResponse": lambda _, nodes: [nodes[1]],
            "String": lambda _, nodes: str(nodes).strip(),
        }
        return actions

    def interpret_result(self, result):
        return result
    
    def parse(self, text, tdf: TrainingDataFormat = None) -> TrainingDataFormat:
        tdf = TrainingDataFormat() if tdf is None else copy.deepcopy(tdf)
        final_tdf : TrainingDataFormat = self.run(text, tdf)
        if self.state_delta:
            final_tdf.end_goals = get_full_state(tdf.start_goals, final_tdf.end_goals)
        return final_tdf

if __name__ == "__main__":
    next_proof_state = """
[STATES]
[DESCRIPTION]
succesfully added the hypothesis to the context
[STATE] 0
algb_mul t (algb_add e c) = algb_add (algb_mul t e) (algb_mul t c)
[HYPOTHESES]
[HYPOTHESIS] c : algb
[HYPOTHESIS] c : algb
[STATE] 1
algb_mul t (algb_add t c) = algb_add (algb_mul t t) (algb_mul t c)
[HYPOTHESES]
[HYPOTHESIS] c : algb
[HYPOTHESIS] t : algb
[END]
"""
    next_proof_state_res = ProofModelPredGrammar().parse(next_proof_state)
    print(next_proof_state_res)

    start_proof_state = """
[STATES]
[DESCRIPTION]
succesfully added the hypothesis to the context
[STATE] 0
algb_add a e = a
[HYPOTHESES]
[HYPOTHESIS] a : algb
[STATE] 1
algb_add e a = a
[HYPOTHESES]
[HYPOTHESIS] a : algb

[PROOFSTEP]
apply algb_add_comm.
[END]
"""
    start_proof_state_res = ProofModelGrammar().parse(start_proof_state)
    print(start_proof_state_res)

    delta_proof_state = """
[STATES]
[DESCRIPTION]
succesfully added the hypothesis to the context
[STATE] 0
DITTO 1
[END]
"""
    next_proof_state_res = ProofModelPredGrammar(state_delta=True).parse(delta_proof_state, start_proof_state_res)
    print(next_proof_state_res)
    print(ProofModelGrammar().format_as_per_grammar(start_proof_state_res))
    print()
    print(ProofModelPredGrammar().format_as_per_grammar(next_proof_state_res))
    actual_proof_state = """
[STATES]
[STATE] 0
is_not_temp s -> is_not_temp d -> is_not_temp d
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 1
In (s, d) (mu1 ++ (s, d) :: mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 2
move_no_temp nil
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 3
is_path ((s, d) :: nil)
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 4
(is_mill mu1 /\ is_mill mu2 /\ dests_disjoint mu1 mu2) /\
((is_mill sigma /\ ~ In d (dests sigma)) /\
 r <> d /\ ~ In r (dests sigma)) /\
((dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\ ~ In r (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In d (dests mu2)) /\ ~ In r (dests mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 5
move_no_temp (mu1 ++ mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 6
temp_last ((d, r) :: (s, d) :: sigma)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 7
is_path ((d, r) :: (s, d) :: sigma)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 8
is_mill mu /\
(is_mill sigma /\
 (is_mill nil /\ ~ In d (dests nil)) /\
 dests_disjoint sigma nil /\ ~ In d (dests sigma)) /\
dests_disjoint mu sigma /\ dests_disjoint mu nil /\ ~ In d (dests mu)
[HYPOTHESES]
[HYPOTHESIS] D : is_path (sigma ++ (s, d) :: nil)
[HYPOTHESIS] C : temp_last (sigma ++ (s, d) :: nil)
[HYPOTHESIS] B : move_no_temp mu
[HYPOTHESIS] A : is_mill mu /\
(is_mill sigma /\
 (is_mill nil /\ ~ In d (dests nil)) /\
 dests_disjoint sigma nil /\ ~ In d (dests sigma)) /\
dests_disjoint mu sigma /\ dests_disjoint mu nil /\ ~ In d (dests mu)
[HYPOTHESIS] WF : state_wf (State mu (sigma ++ (s, d) :: nil) tau)
[HYPOTHESIS] tau : moves

[PROOFSTEP]
auto.
[END]
"""
    state_res = ProofModelGrammar().parse(actual_proof_state)
    next_state = """
[STATES]
[STATE] 0
In (s, d) (mu1 ++ (s, d) :: mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 1
move_no_temp nil
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 2
is_path ((s, d) :: nil)
[HYPOTHESES]
[HYPOTHESIS] D : is_path nil
[HYPOTHESIS] C : temp_last nil
[HYPOTHESIS] B : move_no_temp (mu1 ++ (s, d) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In d (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In d (dests mu1)) /\
is_mill nil /\
dests_disjoint mu1 nil /\ dests_disjoint mu2 nil /\ ~ In d (dests nil)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (s, d) :: mu2) nil tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] s,d : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 3
(is_mill mu1 /\ is_mill mu2 /\ dests_disjoint mu1 mu2) /\
((is_mill sigma /\ ~ In d (dests sigma)) /\
 r <> d /\ ~ In r (dests sigma)) /\
((dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\ ~ In r (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In d (dests mu2)) /\ ~ In r (dests mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 4
move_no_temp (mu1 ++ mu2)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 5
temp_last ((d, r) :: (s, d) :: sigma)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 6
is_path ((d, r) :: (s, d) :: sigma)
[HYPOTHESES]
[HYPOTHESIS] D : is_path ((s, d) :: sigma)
[HYPOTHESIS] C : temp_last ((s, d) :: sigma)
[HYPOTHESIS] B : move_no_temp (mu1 ++ (d, r) :: mu2)
[HYPOTHESIS] A : (is_mill mu1 /\
 (is_mill mu2 /\ ~ In r (dests mu2)) /\
 dests_disjoint mu1 mu2 /\ ~ In r (dests mu1)) /\
(is_mill sigma /\ ~ In d (dests sigma)) /\
(dests_disjoint mu1 sigma /\ ~ In d (dests mu1)) /\
(dests_disjoint mu2 sigma /\ ~ In r (dests sigma)) /\
d <> r /\ ~ In d (dests mu2)
[HYPOTHESIS] WF : state_wf (State (mu1 ++ (d, r) :: mu2) ((s, d) :: sigma) tau)
[HYPOTHESIS] tau : moves
[HYPOTHESIS] sigma : list (reg * reg)
[HYPOTHESIS] s : reg
[HYPOTHESIS] mu2 : list (reg * reg)
[HYPOTHESIS] d,r : reg
[HYPOTHESIS] mu1 : list (reg * reg)
[HYPOTHESIS] val : Type
[HYPOTHESIS] temp : reg -> reg
[HYPOTHESIS] reg_eq : forall r1 r2 : reg, {r1 = r2} + {r1 <> r2}
[HYPOTHESIS] reg : Type
[STATE] 7
is_mill mu /\
(is_mill sigma /\
 (is_mill nil /\ ~ In d (dests nil)) /\
 dests_disjoint sigma nil /\ ~ In d (dests sigma)) /\
dests_disjoint mu sigma /\ dests_disjoint mu nil /\ ~ In d (dests mu)
[HYPOTHESES]
[HYPOTHESIS] D : is_path (sigma ++ (s, d) :: nil)
[HYPOTHESIS] C : temp_last (sigma ++ (s, d) :: nil)
[HYPOTHESIS] B : move_no_temp mu
[HYPOTHESIS] A : is_mill mu /\
(is_mill sigma /\
 (is_mill nil /\ ~ In d (dests nil)) /\
 dests_disjoint sigma nil /\ ~ In d (dests sigma)) /\
dests_disjoint mu sigma /\ dests_disjoint mu nil /\ ~ In d (dests mu)
[HYPOTHESIS] WF : state_wf (State mu (sigma ++ (s, d) :: nil) tau)
[HYPOTHESIS] tau : moves
[END]
"""
    next_state_res = ProofModelPredGrammar().parse(next_state, state_res)
    formatted_out = ProofModelPredGrammar(state_delta=True).format_as_per_grammar(next_state_res)

    actual_proof_state = """
[STATES]
[STATE] 0
forall n m : nat, S n + m = S (n + m)
[END]
"""
    state_res = ProofModelGrammar().parse(actual_proof_state)
    print(state_res)
    next_state = """
[PROOFSTEP]
intros.

[STATES]
[STATE] 0
S n + m = S (n + m)
[HYPOTHESES]
[HYPOTHESIS] n,m : nat
[END]
"""
    next_state_res = ProofModelPredGrammar(state_delta=True).parse(next_state, state_res)
    formatted_out = ProofModelPredGrammar(state_delta=True).format_as_per_grammar(next_state_res)
    print(formatted_out)
    print(next_state_res)