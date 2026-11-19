"""Frontier-CS file submissions from a persistent mini-swe-agent shell."""

import shlex
from dataclasses import replace

from minisweagent.agents.default import FormatError, Submitted

from agentic_critic.fcs_env import COMPACT_PROMPT, FCS_REASONING_EFFORT, FCS_SUMMARY_EFFORT, format_feedback, summary_text
from agentic_critic.terminal_env import SHELL_INSTRUCTIONS, TerminalAgent

_SUBMIT_EXAMPLE = "```bash\n" + r"\submit main.cpp" + "\n```"


class FcsWorkspaceAgent(TerminalAgent):
    def __init__(
        self,
        *,
        tokenizer,
        decode,
        execute,
        submit,
        config,
        max_submissions,
        max_turns,
        reasoning_effort=FCS_REASONING_EFFORT,
    ):
        if max_submissions < 1 or max_turns < 0:
            raise ValueError("FCS requires a positive submission limit and a nonnegative turn limit")
        super().__init__(tokenizer=tokenizer, decode=decode, execute=self._command, config=replace(config, max_steps=max_turns or None))
        self.config.format_error_template = (
            "Use exactly one bash block. To submit your current C++ file to the judge, send:\n" + _SUBMIT_EXAMPLE
        )
        self.shell = execute
        self.submit = submit
        self.max_submissions = max_submissions
        self.reasoning_effort = reasoning_effort

    def _template_options(self, kind):
        summary = kind == "summary"
        return {
            "enable_thinking": self.options.fcs_summary_thinking if summary else self.options.enable_thinking,
            "reasoning_effort": FCS_SUMMARY_EFFORT if summary else self.reasoning_effort,
        }

    def _initial_messages(self, task):
        budget = (
            f"You have {self.options.max_steps} actions in total, including edits, tests and submissions."
            if self.options.max_steps is not None
            else f"You have a total generation budget of {self.options.max_total_tokens:,} tokens."
        )
        system = (
            f"{SHELL_INSTRUCTIONS}\n\n{budget} You may submit up to {self.max_submissions} times. The highest judge score counts. "
            "Submit a first runnable solution as soon as it is ready, then use the feedback to improve it. "
            "Submit the saved file in a separate action, exactly like this (change the path if needed):\n"
            + _SUBMIT_EXAMPLE
        )
        instruction = (
            "Solve the problem in your persistent shell workspace. Write your C++ program to a file. "
            "You can compile it with g++, run it on small inputs and inspect the real output."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": instruction + "\n\n" + task}]

    def has_finished(self, output):
        try:
            super().has_finished(output)
        except Submitted:
            if self.trace.submission_scores:
                raise
            raise FormatError("There is no judge submission yet. Submit your saved file:\n" + _SUBMIT_EXAMPLE)

    def _summary_request(self):
        observation = self.messages[-1]
        return self.messages[:-1] + [{**observation, "content": observation["content"] + "\n\n" + COMPACT_PROMPT}]

    def _summary_resume(self, summary):
        if self.options.fcs_summary_thinking:
            summary = summary_text(summary)
        return super()._summary_resume(summary)

    def _action_content(self, response):
        if response.finish_reason == "length":
            raise FormatError("The action reached its token limit. Issue a shorter bash command.")
        if self.options.enable_thinking and "</think>" not in response.text:
            call = self.trace.calls[-1]
            start = call["token_span"][0]
            prompt_tail = self.trace.segments[call["segment"]].tokens[max(0, start - 16) : start]
            if self.tokenizer.decode(prompt_tail).rstrip().endswith("<think>"):
                raise FormatError("Finish the reasoning block before issuing a bash command.")
        return super()._action_content(response)

    async def _command(self, command, timeout):
        # The agent sends one command at a time. Read submitted files inside the
        # same sandbox as bash; only the host-side submitter has judge access.
        words = command.strip().split(maxsplit=1)
        if not words or words[0] not in (r"\submit", "submit"):
            return await self.shell(command, timeout)
        try:
            paths = shlex.split(words[1]) if len(words) == 2 else []
        except ValueError:
            paths = []
        if len(paths) != 1:
            return {"returncode": 2, "output": r"Usage: \submit <source-file>"}
        file = shlex.quote(paths[0])
        result = await self.shell(f"test -f {file} && cat -- {file}", timeout)
        if result["returncode"]:
            return {"returncode": result["returncode"], "output": f"Cannot read source file: {paths[0]}"}
        verdict = await self.submit(result["output"])
        scores = self.trace.submission_scores
        scores.append(verdict.score)
        self.trace.calls[-1]["action_status"] = "submitted"
        if len(scores) == self.max_submissions:
            raise Submitted()
        return {"returncode": 0, "output": format_feedback(verdict, len(scores), self.max_submissions, max(scores))}
