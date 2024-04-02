import typing
import copy
from enum import Enum
from itp_interface.tools.training_data_format import TrainingDataFormat, Goal
try:
    from .grammars.grammar import Grammar
except ImportError:
    from grammars.grammar import Grammar

class ProofModelGrammar(Grammar):
    grammar = """
Prog:
  StatesStr ProofState ProofStep End;
ProofState:
  Description String States
| States;
States:
  State
| State States
| EMPTY;
State:
 StateStr String HypsResponses;
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


terminals
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

    def __init__(self):
        recognizers = {
            'String': ProofModelGrammar.before_keyword
        }
        super(ProofModelGrammar, self).__init__(ProofModelGrammar.grammar, ProofModelGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, training_data_format: TrainingDataFormat, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            ProofModelGrammar.Keywords.STATE : [],
            ProofModelGrammar.Keywords.PROOFSTEP : [],
        }
        lines_order = [
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
            new_line = f"{ProofModelGrammar.Keywords.STATE}"
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            new_line = str(goal.goal)
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
            if len(goal.hypotheses) > 0:
                new_line = f"{ProofModelGrammar.Keywords.HYPOTHESES}"
                lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
                for hyp in goal.hypotheses:
                    new_line = f"{ProofModelGrammar.Keywords.HYPOTHESIS} {hyp}"
                    lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
        if len(training_data_format.proof_steps) > 0:
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


class ProofModelPredGrammar(Grammar):
    grammar = """
Prog:
  StatesStr ProofState End;
ProofState:
  Description String States
| States;
States:
  State
| State States
| EMPTY;
State:
 StateStr String HypsResponses;
HypsResponses:
    Hyps HypResponses
|   EMPTY;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;

terminals
StateStr: "[STATE]";
StatesStr: "[STATES]";
Hyps: "[HYPOTHESES]";
Hyp: "[HYPOTHESIS]";
End: "[END]";
Description: "[DESCRIPTION]";
String:;
"""

    def __init__(self):
        recognizers = {
            'String': ProofModelGrammar.before_keyword
        }
        super(ProofModelPredGrammar, self).__init__(ProofModelPredGrammar.grammar, ProofModelGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, training_data_format: TrainingDataFormat, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            ProofModelGrammar.Keywords.STATE : []
        }
        lines_order = [
            ProofModelGrammar.Keywords.STATE
        ]
        priority_order_lo_hi = [
            ProofModelGrammar.Keywords.STATE,
        ]
        lines_map[ProofModelGrammar.Keywords.STATE] = [str(ProofModelGrammar.Keywords.STATES)]
        if training_data_format.goal_description is not None:
            new_line = f"{ProofModelGrammar.Keywords.DESCRIPTION}\n{training_data_format.goal_description}\n"
            lines_map[ProofModelGrammar.Keywords.STATE].append(new_line)
        for _, goal in enumerate(training_data_format.end_goals):
            new_line = f"{ProofModelGrammar.Keywords.STATE}"
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


    def _parse_expr(self, rule_type, nodes, context: TrainingDataFormat):
        if rule_type == ProofModelGrammar.Keywords.HYPOTHESIS:
            hyps = []
            for node in nodes:
                hyps.extend(node)
            return hyps
        elif rule_type == ProofModelGrammar.Keywords.HYPOTHESES:
            return nodes[1]
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
            if len(nodes) > 1:
                context.goal_description = nodes[1].strip() if nodes[1] is not None else None
                context.end_goals = nodes[2]
            else:
                context.end_goals = nodes[0]
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
        return self.run(text, tdf)

if __name__ == "__main__":
    next_proof_state = """
[STATES]
[DESCRIPTION]
succesfully added the hypothesis to the context
[STATE]
algb_mul t (algb_add e c) = algb_add (algb_mul t e) (algb_mul t c)
[HYPOTHESES]
[HYPOTHESIS] c : algb
[HYPOTHESIS] c : algb
[STATE]
algb_mul t (algb_add t c) = algb_add (algb_mul t t) (algb_mul t c)
[HYPOTHESES]
[HYPOTHESIS] c : algb
[HYPOTHESIS] t : algb
[END]
"""
    next_proof_state_res = ProofModelPredGrammar().parse(next_proof_state)
    print(next_proof_state_res)