from __future__ import annotations

import unittest

from app.ai.reasoning import FinalAnswerFilter, sanitize_final_text


class ReasoningFilterTests(unittest.TestCase):
    def test_unmarked_final_answer_streams_normally(self):
        parser = FinalAnswerFilter()
        self.assertEqual(parser.feed("The final "), "")
        self.assertEqual(parser.feed("answer is 42."), "")
        self.assertEqual(parser.finish(), "The final answer is 42.")

    def test_open_and_close_think_tags_split_across_chunks_are_withheld(self):
        parser = FinalAnswerFilter()
        visible = []
        for chunk in (
            "<thi", "nk>First thought: keep this private.",
            "</thi", "nk>Final answer: 42.",
        ):
            visible.append(parser.feed(chunk))
        visible.append(parser.finish())
        result = "".join(visible)
        self.assertEqual(result, "Final answer: 42.")
        self.assertNotIn("First thought", result)

    def test_unclosed_think_block_and_partial_marker_are_discarded(self):
        parser = FinalAnswerFilter()
        self.assertEqual(parser.feed("<think>private reasoning without a closer"), "")
        self.assertEqual(parser.finish(), "")

        parser = FinalAnswerFilter()
        visible = parser.feed("A complete answer<thi")
        self.assertEqual(visible + parser.finish(), "A complete answer")

    def test_unmatched_closing_marker_discards_its_preceding_malformed_prefix(self):
        parser = FinalAnswerFilter()
        self.assertEqual(parser.feed("The user just asked for private reasoning"), "")
        self.assertEqual(parser.feed("</think>Visible answer"), "")
        self.assertEqual(parser.finish(), "Visible answer")

    def test_reasoning_before_late_closing_marker_never_leaks(self):
        parser = FinalAnswerFilter()
        self.assertEqual(parser.feed("First thought: The user just asked"), "")
        self.assertEqual(parser.feed(" for a greeting."), "")
        self.assertEqual(parser.feed("</think>Hi, I am well."), "")
        result = parser.finish()
        self.assertEqual(result, "Hi, I am well.")
        self.assertNotIn("First thought", result)
        self.assertNotIn("</think>", result)

    def test_historical_text_sanitization_removes_marked_reasoning(self):
        stored = "<think>First thought. The user just asked about X.</think>Final answer."
        self.assertEqual(sanitize_final_text(stored), "Final answer.")


if __name__ == "__main__":
    unittest.main()
