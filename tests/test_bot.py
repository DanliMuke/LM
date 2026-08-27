import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import bot


class FakeTranscriptions:
    def __init__(self, response):
        self.response = response
        self.params = None

    def create(self, *, file, **params):
        self.params = params
        self.file_was_open = not file.closed
        return self.response


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(output_text=self.output_text)


class FakeTelegramMessage:
    def __init__(self):
        self.message_id = 42
        self.voice = None
        self.audio = None
        self.video_note = None
        self.document = None
        self.forwarded_to = []
        self.replies = []

    async def forward(self, chat_id):
        self.forwarded_to.append(chat_id)

    async def reply_text(self, text):
        self.replies.append(text)


class BotTests(unittest.TestCase):
    def test_model_validation_rejects_text_model(self):
        self.assertFalse(bot.is_supported_transcription_model("gpt-5.6-luna"))
        self.assertTrue(bot.is_supported_transcription_model("gpt-4o-mini-transcribe"))

    def test_diarize_model_does_not_request_logprobs(self):
        self.assertFalse(bot.supports_logprobs("gpt-4o-transcribe-diarize"))
        self.assertTrue(bot.supports_logprobs("gpt-4o-mini-transcribe"))

    def test_confidence_uses_geometric_mean(self):
        confidence = bot.confidence_from_logprobs(
            [
                {"logprob": 0.0},
                SimpleNamespace(logprob=-1.0),
            ]
        )
        self.assertAlmostEqual(confidence, 0.6065306597)

    def test_low_confidence_text_is_not_rewritten(self):
        original = "неуверенно распознанный текст"
        output = bot.build_user_output(
            bot.TranscriptionResult(text=original, confidence=0.20)
        )
        self.assertIn("без исправлений и дополнений", output)
        self.assertTrue(output.endswith(original))

    def test_semantic_validation_accepts_only_formatting_changes(self):
        source = "иван заплатил 12,5 ₽ 01.08.2026"
        formatted = "Иван заплатил 12,5 ₽. 01.08.2026."
        changed_amount = "Иван заплатил 15,5 ₽. 01.08.2026."

        self.assertTrue(bot.preserves_semantic_tokens(source, formatted))
        self.assertFalse(bot.preserves_semantic_tokens(source, changed_amount))
        self.assertFalse(bot.preserves_semantic_tokens(source, formatted + " ✅"))

    def test_cleanup_uses_one_response_call_and_preserves_raw_text(self):
        source = "иван заплатил 12,5 ₽ сегодня"
        responses = FakeResponses("Иван заплатил 12,5 ₽ сегодня.")
        fake_client = SimpleNamespace(responses=responses)
        previous_client = bot.client
        bot.client = fake_client
        try:
            result = asyncio.run(
                bot.clean_transcript(
                    bot.TranscriptionResult(text=source, confidence=0.95)
                )
            )
        finally:
            bot.client = previous_client

        self.assertTrue(result.applied)
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(responses.calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(responses.calls[0]["reasoning"], {"effort": "none"})
        self.assertFalse(responses.calls[0]["store"])
        output = bot.build_user_output(
            bot.TranscriptionResult(text=source, confidence=0.95),
            result,
        )
        self.assertIn(result.text, output)
        self.assertIn(source, output)

    def test_cleanup_rejects_changed_data_without_retry(self):
        source = "Сумма 1000 ₽"
        responses = FakeResponses("Сумма 10 000 ₽.")
        previous_client = bot.client
        bot.client = SimpleNamespace(responses=responses)
        try:
            result = asyncio.run(
                bot.clean_transcript(
                    bot.TranscriptionResult(text=source, confidence=0.95)
                )
            )
        finally:
            bot.client = previous_client

        self.assertTrue(result.rejected)
        self.assertEqual(result.text, source)
        self.assertEqual(len(responses.calls), 1)

    def test_low_confidence_skips_paid_cleanup_call(self):
        responses = FakeResponses("Этот ответ не должен использоваться")
        previous_client = bot.client
        bot.client = SimpleNamespace(responses=responses)
        try:
            result = asyncio.run(
                bot.clean_transcript(
                    bot.TranscriptionResult(text="неясный текст", confidence=0.10)
                )
            )
        finally:
            bot.client = previous_client

        self.assertEqual(result.text, "неясный текст")
        self.assertEqual(len(responses.calls), 0)

    def test_automatic_api_retries_are_disabled_by_default(self):
        self.assertEqual(bot.OPENAI_MAX_RETRIES, 0)

    def test_private_message_is_forwarded_to_manager_group(self):
        message = FakeTelegramMessage()
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(id=123),
            effective_chat=SimpleNamespace(type="private"),
        )
        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = -1001234567890
        try:
            asyncio.run(bot.handle_private_message(update, None))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(message.forwarded_to, [-1001234567890])
        self.assertEqual(message.replies, ["Сообщение передано менеджерам."])

    def test_group_message_is_ignored(self):
        message = FakeTelegramMessage()
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(id=123),
            effective_chat=SimpleNamespace(type="group"),
        )
        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = -1001234567890
        try:
            asyncio.run(bot.handle_private_message(update, None))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(message.forwarded_to, [])
        self.assertEqual(message.replies, [])

    def test_transcription_request_is_deterministic_and_asks_for_logprobs(self):
        response = SimpleNamespace(
            text="  точный текст  ",
            logprobs=[SimpleNamespace(logprob=-0.1)],
        )
        transcriptions = FakeTranscriptions(response)
        fake_client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=transcriptions)
        )
        previous_client = bot.client
        bot.client = fake_client
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as audio_file:
                audio_file.write(b"audio")
                temp_path = Path(audio_file.name)
            result = asyncio.run(bot.transcribe_file(temp_path))
        finally:
            bot.client = previous_client
            if temp_path:
                temp_path.unlink(missing_ok=True)

        self.assertEqual(result.text, "точный текст")
        self.assertTrue(transcriptions.file_was_open)
        self.assertEqual(transcriptions.params["temperature"], 0)
        self.assertEqual(transcriptions.params["response_format"], "json")
        self.assertEqual(transcriptions.params["include"], ["logprobs"])

    def test_long_text_is_split_without_losing_content(self):
        text = "слово " * 1000
        chunks = bot.split_text(text, limit=100)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text.strip())


if __name__ == "__main__":
    unittest.main()
