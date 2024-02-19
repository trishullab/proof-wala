import typing
from enum import Enum
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.training_data_format import TrainingDataFormat
try:
    from .grammars.grammar import Grammar
except ImportError:
    from grammars.grammar import Grammar

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

class CoqGptResponseActions(object):
    GOALS = "[GOALS]"
    ERROR = "[ERROR]"
    ERROR_MESSAGE = "[ERROR MESSAGE]"

@dataclass_json
@dataclass
class CoqGptResponse(object):
    action : str = CoqGptResponseActions.GOALS
    success: bool = True
    message: str = ""
    steps: typing.List[str] = field(default_factory=list)
    incorrect_steps: typing.List[str] = field(default_factory=list)
    error_message: typing.Optional[str] = None
    last_step: typing.Optional[str] = None
    training_data_format: typing.Optional[TrainingDataFormat] = None
    informal_proof: typing.Optional[str] = None
    informal_theorem: typing.Optional[str] = None

class CoqGPTResponseDfsGrammar(Grammar):
    grammar = """
Prog:
  GoalsResponse
| ErrorResponse
| String Prog
| Prog String Prog;
ErrorResponse:
   Error String End
|  Error ErrorString End;
GoalsResponse:
  Goals Description String GoalResponses StepsResponses IncorrectStepsResponses LastResponse End
| Goals GoalResponses StepsResponses IncorrectStepsResponses LastResponse End;
GoalResponses:
  GoalResponse
| GoalResponse GoalResponses
| EMPTY;
GoalResponse:
 Goal int String HypsResponses DfnsResponses ThmsResponses;
DfnsResponses:
    Dfns int DfnResponses
|   EMPTY;
DfnResponses:
    DfnResponse
|   DfnResponse DfnResponses;
DfnResponse:
    Dfn String;
ThmsResponses:
    Thms int ThmResponses
|   EMPTY;
ThmResponses:
    ThmResponse
|   ThmResponse ThmResponses;
ThmResponse:
    Thm String;
HypsResponses:
    Hyps int HypResponses
|   EMPTY;
HypResponses:
  HypResponse
| HypResponse HypResponses;
HypResponse:
  Hyp String;
IncorrectStepsResponses:
    IncrctStps StepResponses
|   EMPTY;
LastResponse:
    LastStep String ErrorMessage String
|   LastStep String Success
|   EMPTY;
StepsResponses:
    Stps StepResponses
|   EMPTY;
StepResponses:
    StepResponse
|   StepResponse StepResponses;
StepResponse:
    Stp String;


terminals
int: /\d+/;
Goals: "[GOALS]";
Goal: "[GOAL]";
Hyps: "[HYPOTHESES]";
Hyp: "[HYPOTHESIS]";
Stps: "[STEPS]";
Stp: "[STEP]";
IncrctStps: "[INCORRECT STEPS]";
Dfns: "[DEFINITIONS]";
Dfn: "[DEFINITION]";
Thms: "[THEOREMS]";
Thm: "[THEOREM]";
Error: "[ERROR]";
ErrorMessage: "[ERROR MESSAGE]";
Success: "[SUCCESS]";
LastStep: "[LAST STEP]";
End: "[END]";
Description: "[DESCRIPTION]";
String:;
ErrorString:;
"""
    class Keywords(Enum):
        GOALS = "[GOALS]"
        GOAL = "[GOAL]"
        HYPOTHESES = "[HYPOTHESES]"
        HYPOTHESIS = "[HYPOTHESIS]"
        STEPS = "[STEPS]"
        STEP = "[STEP]"
        INCORRECT_STEPS = "[INCORRECT STEPS]"
        DEFINITIONS = "[DEFINITIONS]"
        DEFINITION = "[DEFINITION]"
        THEOREMS = "[THEOREMS]"
        THEOREM = "[THEOREM]"
        ERROR = "[ERROR]"
        ERROR_MESSAGE = "[ERROR MESSAGE]"
        SUCCESS = "[SUCCESS]"
        END = "[END]"
        DESCRIPTION = "[DESCRIPTION]"
        LAST_STEP = "[LAST STEP]"
        INFORMAL_PROOF = "[INFORMAL-PROOF]"
        INFORMAL_THEOREM = "[INFORMAL-THEOREM]"

        def __str__(self) -> str:
            return self.value

    keywords = [keyword.value for keyword in Keywords]

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in CoqGPTResponseDfsGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1
    
    def error_string(text, pos):
        last = pos
        while last < len(text):
            while last < len(text) and text[last] != '[':
                last += 1
            if last < len(text) and text[last:].startswith("[END]") and text[last:].endswith("[END]"):
                return text[pos:last]
            last += 1

    def __init__(self):
        recognizers = {
            'String': CoqGPTResponseDfsGrammar.before_keyword,
            'ErrorString': CoqGPTResponseDfsGrammar.error_string
        }
        super(CoqGPTResponseDfsGrammar, self).__init__(CoqGPTResponseDfsGrammar.grammar, CoqGPTResponseDfsGrammar.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: CoqGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: float = 4.0) -> str:
        # Add algorithm for trimming the right amount of goals, theorems and defintions, steps, etc. based on the max_token_cnt
        char_cnt = int(max_token_cnt * characters_per_token) if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        if coq_gpt_response.action == CoqGptResponseActions.ERROR:
            text = f"{CoqGPTResponseDfsGrammar.Keywords.ERROR}\n{coq_gpt_response.message}\n{CoqGPTResponseDfsGrammar.Keywords.END}"
        elif coq_gpt_response.action == CoqGptResponseActions.GOALS:
            lines_map = {
                CoqGPTResponseDfsGrammar.Keywords.GOALS : [],
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM : [],
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF : [],
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS : [],
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS : [],
                CoqGPTResponseDfsGrammar.Keywords.STEPS : [],
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS : [],
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP : [],
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS : [],
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE : []
            }
            lines_order = [
                CoqGPTResponseDfsGrammar.Keywords.GOALS,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF,
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS,
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS,
                CoqGPTResponseDfsGrammar.Keywords.STEPS,
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS,
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP,
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS,
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE
            ]
            priority_order_lo_hi = [
                CoqGPTResponseDfsGrammar.Keywords.THEOREMS,
                CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS,
                CoqGPTResponseDfsGrammar.Keywords.STEPS,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF,
                CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM,
                CoqGPTResponseDfsGrammar.Keywords.GOALS, # trim down the goals
                CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS,
                CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE,
                CoqGPTResponseDfsGrammar.Keywords.LAST_STEP,
                CoqGPTResponseDfsGrammar.Keywords.SUCCESS,
            ]
            assert coq_gpt_response.training_data_format is not None
            new_line = f"Goals to prove:\n{CoqGPTResponseDfsGrammar.Keywords.GOALS}"
            lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS] = [new_line]
            if coq_gpt_response.training_data_format.goal_description is not None:
                new_line = f"{CoqGPTResponseDfsGrammar.Keywords.DESCRIPTION}\n{coq_gpt_response.training_data_format.goal_description}\n"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
            for i, goal in enumerate(coq_gpt_response.training_data_format.start_goals):
                new_line = f"{CoqGPTResponseDfsGrammar.Keywords.GOAL} {i+1}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                new_line = str(goal.goal)
                lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                if len(goal.hypotheses) > 0:
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESES} {i + 1}"
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                    for hyp in goal.hypotheses:
                        new_line = f"{CoqGPTResponseDfsGrammar.Keywords.HYPOTHESIS} {hyp}"
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.GOALS].append(new_line)
                if len(goal.relevant_defns) > 0 and (k is None or k > 0):
                    dfns = goal.relevant_defns
                    if k is not None:
                        dfns = dfns[:k]
                    dfns = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[dfn.lemma_idx]) for dfn in dfns]
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS} {i + 1}"
                    if len(lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS]) == 0:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS] = [new_line]
                    else:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS].append(new_line)
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.DEFINITIONS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.DEFINITION} {dfn}" for dfn in dfns])
                if len(goal.possible_useful_theorems_external) + len(goal.possible_useful_theorems_local) > 0 and (k is None or k > 0):
                    thms = goal.possible_useful_theorems_local + goal.possible_useful_theorems_external
                    if k is not None:
                        thms = thms[:k]
                    thms = [str(coq_gpt_response.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) for thm in thms]
                    new_line = f"{CoqGPTResponseDfsGrammar.Keywords.THEOREMS} {i + 1}"
                    if len(lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS]) == 0:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS] = [new_line]
                    else:
                        lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS].append(new_line)
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.THEOREMS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.THEOREM} {thm}" for thm in thms])
            if len(coq_gpt_response.steps) > 0:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.STEPS}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.STEPS] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.STEPS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.steps])
            if len(coq_gpt_response.incorrect_steps) > 0:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INCORRECT_STEPS].extend([f"{CoqGPTResponseDfsGrammar.Keywords.STEP} {step}" for step in coq_gpt_response.incorrect_steps])
            if coq_gpt_response.last_step is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.LAST_STEP}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.LAST_STEP] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.LAST_STEP].append(coq_gpt_response.last_step)
                if coq_gpt_response.success:
                    new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.SUCCESS}"
                    lines_map[CoqGPTResponseDfsGrammar.Keywords.SUCCESS] = [new_line]
            if coq_gpt_response.error_message is not None:
                # assert coq_gpt_response.last_step is not None
                # assert not coq_gpt_response.success
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.ERROR_MESSAGE].append(coq_gpt_response.error_message)
            if coq_gpt_response.informal_theorem is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_THEOREM].append(coq_gpt_response.informal_theorem)
            if coq_gpt_response.informal_proof is not None:
                new_line = f"\n{CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF}"
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF] = [new_line]
                lines_map[CoqGPTResponseDfsGrammar.Keywords.INFORMAL_PROOF].append(coq_gpt_response.informal_proof)
            keywords = [keyword for keyword in lines_map.keys()]
            # Convert all the lines under each keyword to a single string
            for keyword in keywords:
                lines_map[keyword] = "\n".join(lines_map[keyword])
            # Frame the first prompt version without any token limit
            text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{CoqGPTResponseDfsGrammar.Keywords.END}"
            
            # Trim the lines based on the max_token_cnt
            if char_cnt is not None and len(text) > char_cnt:
                _idx = 0
                diff = len(text) - char_cnt
                while _idx < len(priority_order_lo_hi) and diff > 0:
                    trim_part = priority_order_lo_hi[_idx]
                    if trim_part in lines_map:
                        if trim_part == CoqGPTResponseDfsGrammar.Keywords.STEPS:
                            if len(lines_map[trim_part]) <= diff:
                                lines_map[trim_part] = ""
                            else:
                                lines_map[trim_part] = lines_map[trim_part][diff:]
                        else:
                            lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                    text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0]) + f"\n{CoqGPTResponseDfsGrammar.Keywords.END}"
                    diff = len(text) - char_cnt
                    _idx += 1
        else:
            raise Exception(f"Invalid action {coq_gpt_response.action}")
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None:
            assert len(text) <= char_cnt, f"Text length {len(text)} is greater than the max token count {char_cnt}. Possibly too few characters per token." +\
            f" characters_per_token = {characters_per_token}, max_token_cnt = {max_token_cnt}"
            # text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text


class CoqGptRequestActions(object):
    RUN_TACTIC = "[RUN TACTIC]"
    GET_DFNS_THMS = "[GET DEFINITIONS AND THEOREMS]"

@dataclass_json
@dataclass
class CoqGptRequest(object):
    action : str = CoqGptRequestActions.RUN_TACTIC
    args: typing.List[str] = field(default_factory=list)

class CoqGPTRequestGrammar(Grammar):
    grammar = """
Prog: 
  RunTacticRequest
| GetDfnsThmsRequest
| String Prog;
GetDfnsThmsRequest:
    GetDfnsThms End;
RunTacticRequest:
    RunTactic StpRequests End;
StpRequests:
  String;

terminals
Stp: "[STEP]";
End: "[END]";
RunTactic: "[RUN TACTIC]";
GetDfnsThms: "[GET DEFINITIONS AND THEOREMS]";
String:;
"""
    keywords = ["[STEP]", "[END]", "[RUN TACTIC]", "[GET DEFINITIONS]"]

    end = "[END]"

    def before_keyword(text, pos):
        last = pos
        while last < len(text):
          while last < len(text) and text[last] != '[':
            last += 1
          if last < len(text):
              for keyword in CoqGPTRequestGrammar.keywords:
                  if text[last:].startswith(keyword):
                      return text[pos:last]
              last += 1

    def __init__(self, enable_defensive_parsing: bool = False):
        recognizers = {
            'String': CoqGPTRequestGrammar.before_keyword
        }
        self.enable_defensive_parsing = enable_defensive_parsing
        super(CoqGPTRequestGrammar, self).__init__(CoqGPTRequestGrammar.grammar, CoqGPTRequestGrammar.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal, nodes, context):
        if nonTerminal == "GetDfnsThmsRequest":
            context.action = CoqGptRequestActions.GET_DFNS_THMS
            context.args = [CoqGptRequestActions.GET_DFNS_THMS[1:-1]]
        elif nonTerminal == "RunTacticRequest":
            assert len(nodes) >= 2
            context.action = CoqGptRequestActions.RUN_TACTIC
            context.args.reverse()
        elif nonTerminal == "StpRequests":
            assert len(nodes) >= 1
            str_node = str(nodes[0]).strip()
            if len(str_node) > 0:
                context.args.append(str_node)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        return context
        
    def get_action(self, inp=None):
        context = CoqGptRequest()
        actions = {
            "Prog": lambda _, nodes: context,
            "GetDfnsThmsRequest": lambda _, nodes: self._parse_expr('GetDfnsThmsRequest', nodes, context),
            "RunTacticRequest": lambda _, nodes: self._parse_expr('RunTacticRequest', nodes, context),
            "StpRequests": lambda _, nodes: self._parse_expr('StpRequests', nodes, context),
            "String": lambda _, nodes: str(nodes) # Since this is always a string
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, CoqGptRequest), f"Result must be a CoqGptRequest. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: CoqGptRequest) -> str:
        if coq_gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
            args = '\n'.join(coq_gpt_request.args)
            return f"{CoqGptRequestActions.RUN_TACTIC}\n{args}\n{CoqGPTRequestGrammar.end}"
        elif coq_gpt_request.action == CoqGptRequestActions.GET_DFNS_THMS:
            return f"{CoqGptRequestActions.GET_DFNS_THMS}\n{CoqGPTRequestGrammar.end}"
        else:
            raise Exception(f"Invalid action {coq_gpt_request.action}")

    def get_openai_request(self, message_response: str) -> typing.Tuple[CoqGptRequest, str]:
        message, finish_reason = message_response
        defensive_parsing = finish_reason != "stop" or self.enable_defensive_parsing
        if defensive_parsing:     
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
    
    def normal_parsing(self, message):
        message += CoqGPTRequestGrammar.end
        result : CoqGptRequest = self.run(message, None)            
        message = self.generate_message_from_gpt_request(result)
        return (result, message)

    def defensive_parsing(self, message):
        start_idx = 0
        end_idx = len(message)
        # Generate all possible sub-strings such that the start_idx is less than end_idx
        idxs = [(s_idx, e_idx) for s_idx in range(start_idx, end_idx) for e_idx in range(end_idx, s_idx, -1)]
        message_temp = message
        message_parsed = False
        for s_idx, e_idx in idxs:
            # This type of robust parsing can be needed in case of some LLMs which
            # don't follow the specified format
            try:
                message_temp = message[s_idx:e_idx]
                if message_temp.endswith(CoqGPTRequestGrammar.end):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(CoqGPTRequestGrammar.end)
                message_temp += f"\n{CoqGPTRequestGrammar.end}"
                result : CoqGptRequest = self.run(message_temp, None)            
                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{CoqGPTRequestGrammar.end}"
            result : CoqGptRequest = self.run(message_temp, None)            
            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)

    def attempt_parsing(self, message):
        # do a greedy correction to ensure that the message is parsable
        idx = len(message)
        exceptions = []
        message_seems_fixable = True
        try:
            # trim any unwanted keywords at the end
            idx = message.rfind('[')
            if idx < 0:
                raise Exception("No opening bracket found, message is not parsable")
            close_idx = message.rfind(']', idx, len(message))
            if close_idx < 0:
                message = message[:idx]
            else:
                idx = len(message)
        except Exception:
            message_seems_fixable = False
            pass
        if message_seems_fixable:    
            attempt = 0
            while idx >= 0:
                try:
                    parsable_message = message[:idx] + f"\n{CoqGPTRequestGrammar.end}"
                    self.compile(parsable_message)
                    break
                except Exception as e:
                    exceptions.append(e)
                    idx = message.rfind('[', 0, idx)
                attempt += 1
            if idx >= 0:
                message = parsable_message
            else:
                raise exceptions[0]
            result : CoqGptRequest = self.run(message, None)
            if result.action == CoqGptRequestActions.RUN_TACTIC and len(result.args) > 1:
                result.args = result.args[:-1] # remove the last tactic as it can be incomplete
        else:
            message += CoqGPTRequestGrammar.end
            result : CoqGptRequest = self.run(message, None)
        message = self.generate_message_from_gpt_request(result)
        return (result, message)

    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(CoqGPTRequestGrammar.end), "Message must end with end token"
            result : CoqGptRequest = self.run(message, None)
            results.extend(result.args)
        return results
    
class FewShotGptCoqKeywords(object):
    PROOF = "Proof."
    QED = "Qed."
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"

class FewShotGptLeanKeywords(object):
    PROOF = "[PROOF]"
    QED = "[END]"
    THEOREM = "[THEOREM]"
    DEFINITION = "[DEFINITION]"
    DEFINITIONS = "[DEFINITIONS]"
    LEMMAS = "[LEMMAS]"
    LEMMA = "[LEMMA]"
    END = "[END]"
    INFORMAL_THEOREM = "[INFORMAL-THEOREM]"
    INFORMAL_PROOF = "[INFORMAL-PROOF]"

@dataclass_json
@dataclass
class FewShotGptRequest(object):
    action : ProofAction
    proof_string : str

@dataclass_json
@dataclass
class FewShotGptResponse(object):
    theorem: str
    defintions: typing.List[str] = field(default_factory=list)
    lemmas: typing.List[str] = field(default_factory=list)
    informal_theorem: typing.Optional[str] = None
    informal_proof: typing.Optional[str] = None

class FewShotGptResponseGrammar(Grammar):
    grammar = f"""
Prog:
  Thm String End
| Thm String DfnsResponses LmsResponses End
| Thm String InfThm String InfPrf String End;
DfnsResponses:
    Dfns DfnResponses
|   EMPTY;
DfnResponses:
    DfnResponse
|   DfnResponse DfnResponses;
DfnResponse:
    Dfn String;
LmsResponses:
    Lms LmResponses
|   EMPTY;
LmResponses:
    LmResponse
|   LmResponse LmResponses;
LmResponse:
    Lm String;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.COQ):
        self.language = language
        if language == ProofAction.Language.COQ:
            self.keywords = [FewShotGptCoqKeywords.THEOREM, FewShotGptCoqKeywords.DEFINITION, FewShotGptCoqKeywords.DEFINITIONS, FewShotGptCoqKeywords.LEMMA, FewShotGptCoqKeywords.LEMMAS, FewShotGptCoqKeywords.END]
        elif language == ProofAction.Language.LEAN:
            self.keywords = [FewShotGptLeanKeywords.THEOREM, FewShotGptLeanKeywords.DEFINITION, FewShotGptLeanKeywords.DEFINITIONS, FewShotGptLeanKeywords.LEMMA, FewShotGptLeanKeywords.LEMMAS, FewShotGptLeanKeywords.END, FewShotGptLeanKeywords.INFORMAL_THEOREM, FewShotGptLeanKeywords.INFORMAL_PROOF]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if language == ProofAction.Language.COQ:
            terminals = f"""
terminals
End: "{FewShotGptCoqKeywords.END}";
Thm: "{FewShotGptCoqKeywords.THEOREM}";
Dfn: "{FewShotGptCoqKeywords.DEFINITION}";
Dfns: "{FewShotGptCoqKeywords.DEFINITIONS}";
Lm: "{FewShotGptCoqKeywords.LEMMA}";
Lms: "{FewShotGptCoqKeywords.LEMMAS}";
String:;
"""
        elif language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
End: "{FewShotGptLeanKeywords.END}";
Thm: "{FewShotGptLeanKeywords.THEOREM}";
Dfn: "{FewShotGptLeanKeywords.DEFINITION}";
Dfns: "{FewShotGptLeanKeywords.DEFINITIONS}";
Lm: "{FewShotGptLeanKeywords.LEMMA}";
Lms: "{FewShotGptLeanKeywords.LEMMAS}"
InfThm: "{FewShotGptLeanKeywords.INFORMAL_THEOREM}"
InfPrf: "{FewShotGptLeanKeywords.INFORMAL_PROOF}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.COQ:
            self.END = FewShotGptCoqKeywords.END
            self.THEOREM = FewShotGptCoqKeywords.THEOREM
            self.DEFINITION = FewShotGptCoqKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptCoqKeywords.DEFINITIONS
            self.LEMMA = FewShotGptCoqKeywords.LEMMA
            self.LEMMAS = FewShotGptCoqKeywords.LEMMAS
        elif language == ProofAction.Language.LEAN:
            self.END = FewShotGptLeanKeywords.END
            self.THEOREM = FewShotGptLeanKeywords.THEOREM
            self.DEFINITION = FewShotGptLeanKeywords.DEFINITION
            self.DEFINITIONS = FewShotGptLeanKeywords.DEFINITIONS
            self.LEMMA = FewShotGptLeanKeywords.LEMMA
            self.LEMMAS = FewShotGptLeanKeywords.LEMMAS
            self.INFORMAL_THEOREM = FewShotGptLeanKeywords.INFORMAL_THEOREM
            self.INFORMAL_PROOF = FewShotGptLeanKeywords.INFORMAL_PROOF
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = FewShotGptResponseGrammar.grammar + terminals
        super(FewShotGptResponseGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: FewShotGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: int = 4) -> str:
        assert coq_gpt_response.theorem is not None
        char_cnt = max_token_cnt * characters_per_token if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            self.THEOREM : [],
            self.DEFINITIONS : [],
            self.LEMMAS : [],
        }
        lines_order = [
            self.THEOREM,
            self.DEFINITIONS,
            self.LEMMAS
        ]
        priority_order_lo_hi = [
            self.LEMMAS,
            self.DEFINITIONS,
            self.THEOREM
        ]
        if self.language == ProofAction.Language.LEAN:
            lines_map[self.INFORMAL_THEOREM] = []
            lines_map[self.INFORMAL_PROOF] = []
            lines_order = [
                self.THEOREM,
                self.INFORMAL_THEOREM,
                self.INFORMAL_PROOF,
                self.DEFINITIONS,
                self.LEMMAS
            ]
            priority_order_lo_hi = [
                self.LEMMAS,
                self.DEFINITIONS,
                self.INFORMAL_PROOF,
                self.INFORMAL_THEOREM,
                self.THEOREM
            ]
        new_line = f"{self.THEOREM}\n{coq_gpt_response.theorem}"
        lines_map[self.THEOREM] = [new_line]

        if len(coq_gpt_response.defintions) > 0:
            lines_map[self.DEFINITIONS] = [f"\n{self.DEFINITIONS}"]
        for idx, dfn in enumerate(coq_gpt_response.defintions):
            if k is not None and idx >= k:
                break
            lines_map[self.DEFINITIONS].append(f"{self.DEFINITION} {dfn}")

        if len(coq_gpt_response.lemmas) > 0:
            lines_map[self.LEMMAS] = [f"\n{self.LEMMAS}"]
        for idx, lm in enumerate(coq_gpt_response.lemmas):
            if k is not None and idx >= k:
                break
            lines_map[self.LEMMAS].append(f"{self.LEMMA} {lm}")
        
        if self.language == ProofAction.Language.LEAN:
            if coq_gpt_response.informal_theorem is not None:
                lines_map[self.INFORMAL_THEOREM] = ["\n" + self.INFORMAL_THEOREM]
                lines_map[self.INFORMAL_THEOREM].append(coq_gpt_response.informal_theorem)
            if coq_gpt_response.informal_proof is not None:
                lines_map[self.INFORMAL_PROOF] = [self.INFORMAL_PROOF]
                lines_map[self.INFORMAL_PROOF].append(coq_gpt_response.informal_proof)

        keywords = [keyword for keyword in lines_map.keys()]
        # Convert all the lines under each keyword to a single string
        for keyword in keywords:
            lines_map[keyword] = "\n".join(lines_map[keyword])
        # Frame the first prompt version without any token limit
        text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
        
        # Trim the lines based on the max_token_cnt
        if char_cnt is not None and len(text) > char_cnt:
            _idx = 0
            diff = len(text) - char_cnt
            while _idx < len(priority_order_lo_hi) and diff > 0:
                trim_part = priority_order_lo_hi[_idx]
                if trim_part in lines_map:
                    lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
                diff = len(text) - char_cnt
                _idx += 1
        text += f"\n\n{self.END}"
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None and len(text) > char_cnt:
            text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text


class FewShotGptRequestGrammar(Grammar):
    grammar = f"""
Prog: 
    Proof String Qed;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.COQ, enable_defensive_parsing: bool = False):
        self.language = language
        if language == ProofAction.Language.COQ:
            self.keywords = [FewShotGptCoqKeywords.PROOF, FewShotGptCoqKeywords.QED]
        elif language == ProofAction.Language.LEAN:
            self.keywords = [FewShotGptLeanKeywords.PROOF, FewShotGptLeanKeywords.QED]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if self.language == ProofAction.Language.COQ:
            terminals = f"""
terminals
Proof: "{FewShotGptCoqKeywords.PROOF}";
Qed: "{FewShotGptCoqKeywords.QED}";
String:;
"""
        elif self.language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
Proof: "{FewShotGptLeanKeywords.PROOF}";
Qed: "{FewShotGptLeanKeywords.QED}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.COQ:
            self.PROOF = FewShotGptCoqKeywords.PROOF
            self.QED = FewShotGptCoqKeywords.QED
        elif language == ProofAction.Language.LEAN:
            self.PROOF = FewShotGptLeanKeywords.PROOF
            self.QED = FewShotGptLeanKeywords.QED
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = FewShotGptRequestGrammar.grammar + terminals
        self.enable_defensive_parsing = enable_defensive_parsing
        super(FewShotGptRequestGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal: str, nodes) -> FewShotGptRequest:
        if nonTerminal == "Prog":
            assert len(nodes) >= 3
            if self.language == ProofAction.Language.COQ:
                actions = str(nodes[1]).strip() + f"\n{self.QED}"
            elif self.language == ProofAction.Language.LEAN:
                actions = str(nodes[1]).strip()
                if actions.startswith("begin"):
                    actions = actions[len("begin"):]
                if not actions.endswith("end"):
                    actions += "end"
            else:
                raise NotImplementedError(f"language {self.language} not supported")
            proof_action = ProofAction(ProofAction.ActionType.RUN_TACTIC, self.language, tactics=[actions])
            return FewShotGptRequest(action=proof_action, proof_string=actions)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        
    def get_action(self, inp=None):
        actions = {
            "Prog": lambda _, nodes: self._parse_expr('Prog', nodes)
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, FewShotGptRequest), f"Result must be a FewShotGptRequest. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: FewShotGptRequest) -> str:
        return f"{self.PROOF}\n{coq_gpt_request.proof_string}"

    def get_openai_request(self, message_response: str) -> typing.Tuple[FewShotGptRequest, str]:
        message, _ = message_response
        if self.enable_defensive_parsing:
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
        
    def normal_parsing(self, message):
        message_temp = message
        message_temp += f"\n{self.QED}"
        result : FewShotGptRequest = self.run(message_temp, None)            
        message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)

    def defensive_parsing(self, message):
        start_idx = 0
        end_idx = len(message)
        # Generate all possible sub-strings such that the start_idx is less than end_idx
        idxs = [(s_idx, e_idx) for s_idx in range(start_idx, end_idx) for e_idx in range(end_idx, s_idx, -1)]
        message_temp = message
        message_parsed = False
        for s_idx, e_idx in idxs:
            # This type of robust parsing can be needed in case of some LLMs which
            # don't follow the specified format
            try:
                message_temp = message[s_idx:e_idx]
                if message_temp.endswith(self.QED):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(self.QED)
                message_temp += f"\n{self.QED}"
                result : FewShotGptRequest = self.run(message_temp, None)            
                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{self.QED}"
            result : FewShotGptRequest = self.run(message_temp, None)            
            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(self.QED), "Message must end with end token"
            result : FewShotGptRequest = self.run(message, None)
            results.extend(result.args)
        return results

class InformalFewShotGptLeanKeywords(object):
    THEOREM = "[THEOREM]"
    PROOF = "[PROOF]"
    END = "[END]"

@dataclass_json
@dataclass
class InformalFewShotGptResponse(object):
    theorem: str

class InformalFewShotGptResponseGrammar(Grammar):
    grammar = f"""
Prog:
  Theorem String End;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.LEAN):
        self.language = language
        if language == ProofAction.Language.LEAN:
            self.keywords = [InformalFewShotGptLeanKeywords.THEOREM, InformalFewShotGptLeanKeywords.END]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
End: "{InformalFewShotGptLeanKeywords.END}";
Theorem: "{InformalFewShotGptLeanKeywords.THEOREM}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.LEAN:
            self.END = InformalFewShotGptLeanKeywords.END
            self.THEOREM = InformalFewShotGptLeanKeywords.THEOREM
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = InformalFewShotGptResponseGrammar.grammar + terminals
        super(InformalFewShotGptResponseGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)
    
    def format_as_per_grammar(self, coq_gpt_response: InformalFewShotGptResponse, k: typing.Optional[int] = None, max_token_cnt: typing.Optional[int] = None, characters_per_token: int = 4) -> str:
        assert coq_gpt_response.theorem is not None
        char_cnt = max_token_cnt * characters_per_token if max_token_cnt is not None else None # 4 is the average length of a token as per OpenAI
        text = ""
        lines_map = {
            self.THEOREM : [],
            self.END : []
        }
        lines_order = [
            self.THEOREM,
            self.END
        ]
        priority_order_lo_hi = [
            self.THEOREM,
            self.END
        ]
        
        new_line = f"{self.THEOREM}\n{coq_gpt_response.theorem}"
        lines_map[self.THEOREM] = [new_line]

        keywords = [keyword for keyword in lines_map.keys()]
        # Convert all the lines under each keyword to a single string
        for keyword in keywords:
            lines_map[keyword] = "\n".join(lines_map[keyword])
        # Frame the first prompt version without any token limit
        text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
        
        # Trim the lines based on the max_token_cnt
        if char_cnt is not None and len(text) > char_cnt:
            _idx = 0
            diff = len(text) - char_cnt
            while _idx < len(priority_order_lo_hi) and diff > 0:
                trim_part = priority_order_lo_hi[_idx]
                if trim_part in lines_map:
                    lines_map[trim_part] = lines_map[trim_part][:-diff] # Trim everything except the STEPS from the end
                text = "\n".join([lines_map[keyword] for keyword in lines_order if keyword in lines_map if len(lines_map[keyword]) > 0])
                diff = len(text) - char_cnt
                _idx += 1
        text += f"\n\n{self.END}"
        # verify that the text is valid as per grammar by compiling it
        # self.compile(text)
        if char_cnt is not None and len(text) > char_cnt:
            text = text[:char_cnt] # Just trim the text from the end because no trimming strategy has worked out
        return text


class InformalFewShotGptRequestGrammar(Grammar):
    grammar = f"""
Prog: 
    Proof String End;
"""

    def before_keyword(self, text, pos):
        last = pos
        while last < len(text):
            for keyword in self.keywords:
                if text[last:].startswith(keyword):
                    return text[pos:last]
            last += 1

    def __init__(self, language: ProofAction.Language = ProofAction.Language.LEAN, enable_defensive_parsing: bool = False):
        self.language = language
        if language == ProofAction.Language.LEAN:
            self.keywords = [InformalFewShotGptLeanKeywords.PROOF, InformalFewShotGptLeanKeywords.END]
        else:
            raise NotImplementedError(f"language {language} not supported")
        recognizers = {
            'String': self.before_keyword
        }
        if self.language == ProofAction.Language.LEAN:
            terminals = f"""
terminals
Proof: "{InformalFewShotGptLeanKeywords.PROOF}";
End: "{InformalFewShotGptLeanKeywords.END}";
String:;
"""
        else:
            raise NotImplementedError(f"language {language} not supported")
        if language == ProofAction.Language.LEAN:
            self.PROOF = InformalFewShotGptLeanKeywords.PROOF
            self.END = InformalFewShotGptLeanKeywords.END
        else:
            raise NotImplementedError(f"language {language} not supported")
        grammar = InformalFewShotGptRequestGrammar.grammar + terminals
        self.enable_defensive_parsing = enable_defensive_parsing
        super(InformalFewShotGptRequestGrammar, self).__init__(grammar, self.keywords, recognizers=recognizers)

    def _parse_expr(self, nonTerminal: str, nodes) -> ProofAction:
        if nonTerminal == "Prog":
            assert len(nodes) >= 3
            proof = str(nodes[1]).strip()
            return ProofAction(action_type=ProofAction.ActionType.INFORMAL, language=self.language, proof=proof)
        else:
            raise Exception(f"Unknown non-terminal {nonTerminal}")
        
    def get_action(self, inp=None):
        actions = {
            "Prog": lambda _, nodes: self._parse_expr('Prog', nodes)
        }
        return actions
    
    def interpret_result(self, result):
        assert isinstance(result, ProofAction), f"Result must be a ProofAction. Got {type(result)}"
        return result
    
    def generate_message_from_gpt_request(self, coq_gpt_request: ProofAction) -> str:
        assert coq_gpt_request.action_type == ProofAction.ActionType.INFORMAL, f"action_type must be {ProofAction.ActionType.INFORMAL}, not {coq_gpt_request.action_type}"
        return f"{self.PROOF}\n{coq_gpt_request.kwargs['proof']}\n{self.END}"

    def get_openai_request(self, message_response: str) -> typing.Tuple[ProofAction, str]:
        message, _ = message_response
        if self.enable_defensive_parsing:
            return self.defensive_parsing(message)
        else:
            return self.normal_parsing(message)
    
    def normal_parsing(self, message):
        message_temp = message
        message_temp += f"\n{self.END}"
        result : ProofAction = self.run(message_temp, None)            
        message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)

    def defensive_parsing(self, message):
        start_idx = 0
        end_idx = len(message)
        # Generate all possible sub-strings such that the start_idx is less than end_idx
        idxs = [(s_idx, e_idx) for s_idx in range(start_idx, end_idx) for e_idx in range(end_idx, s_idx, -1)]
        message_temp = message
        message_parsed = False
        for s_idx, e_idx in idxs:
            # This type of robust parsing can be needed in case of some LLMs which
            # don't follow the specified format
            try:
                message_temp = message[s_idx:e_idx]
                if message_temp.endswith(self.END):
                    # Just in case the LLM doesn't remove the stop token
                    message_temp = message_temp.strip(self.END)
                message_temp += f"\n{self.END}"
                result : ProofAction = self.run(message_temp, None)            
                message_temp = self.generate_message_from_gpt_request(result)
                message_parsed = True
            except:
                message_parsed = False
            if message_parsed:
                break
        if not message_parsed:
            message_temp = message[start_idx:end_idx]
            message_temp += f"\n{self.END}"
            result : ProofAction = self.run(message_temp, None)            
            message_temp = self.generate_message_from_gpt_request(result)
        return (result, message_temp)
    
    def parse_request_to_args(self, messages: typing.List[str]) -> typing.List[str]:
        results : typing.List[str] = []
        for message in messages:
            assert message.endswith(self.END), "Message must end with end token"
            result : ProofAction = self.run(message, None)
            results.extend(result.args)
        return results