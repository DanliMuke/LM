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


class FakeManagerBot:
    def __init__(self):
        self.sent_messages = []
        self.edited_captions = []

    async def send_message(self, *, chat_id, text):
        self.sent_messages.append({"chat_id": chat_id, "text": text})

    async def edit_message_caption(self, *, chat_id, message_id, caption):
        self.edited_captions.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "caption": caption,
            }
        )


class FakeTelegramFile:
    async def download_to_drive(self, *, custom_path):
        Path(custom_path).write_bytes(b"audio")


class FakeAudioPayload:
    file_size = 5

    async def get_file(self):
        return FakeTelegramFile()


class FakeTelegramMessage:
    def __init__(self, *, text=None):
        self.message_id = 42
        self.text = text
        self.caption = None
        self.voice = None
        self.audio = None
        self.video_note = None
        self.document = None
        self.forwarded_to = []
        self.copies = []
        self.replies = []

    async def forward(self, chat_id):
        self.forwarded_to.append(chat_id)

    async def copy(self, *, chat_id, caption=None):
        self.copies.append({"chat_id": chat_id, "caption": caption})
        return SimpleNamespace(message_id=9001)

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

    def test_chat_id_command_works_without_manager_chat_configuration(self):
        message = FakeTelegramMessage()
        update = SimpleNamespace(
            message=message,
            effective_chat=SimpleNamespace(id=-1001234567890),
        )
        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = None
        try:
            asyncio.run(bot.chat_id_command(update, None))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(message.replies, ["ID этого чата: -1001234567890"])

    def test_private_message_is_not_processed_before_manager_group_setup(self):
        message = FakeTelegramMessage(text="Здравствуйте")
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(id=123),
            effective_chat=SimpleNamespace(type="private"),
        )
        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = None
        try:
            asyncio.run(bot.handle_private_message(update, None))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(message.forwarded_to, [])
        self.assertEqual(message.copies, [])
        self.assertEqual(message.replies, [])

    def test_private_text_is_sent_only_to_manager_group_with_author(self):
        message = FakeTelegramMessage(text="Здравствуйте")
        manager_bot = FakeManagerBot()
        context = SimpleNamespace(bot=manager_bot)
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(
                id=123,
                full_name="Иван Иванов",
                username="ivan",
            ),
            effective_chat=SimpleNamespace(type="private"),
        )
        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = -1001234567890
        try:
            asyncio.run(bot.handle_private_message(update, context))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(message.replies, [])
        self.assertEqual(len(manager_bot.sent_messages), 1)
        manager_text = manager_bot.sent_messages[0]["text"]
        self.assertIn("Автор: Иван Иванов (@ivan, ID: 123)", manager_text)
        self.assertIn("Здравствуйте", manager_text)

    def test_voice_and_transcript_are_one_manager_message_with_author(self):
        message = FakeTelegramMessage()
        message.voice = FakeAudioPayload()
        manager_bot = FakeManagerBot()
        context = SimpleNamespace(bot=manager_bot)
        update = SimpleNamespace(
            message=message,
            effective_user=SimpleNamespace(
                id=123,
                full_name="Иван Иванов",
                username="ivan",
            ),
            effective_chat=SimpleNamespace(type="private"),
        )

        async def fake_transcribe(_path):
            return bot.TranscriptionResult(text="добрый день", confidence=0.95)

        async def fake_cleanup(_result):
            return bot.CleanupResult(text="Добрый день.", applied=True)

        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        previous_transcribe = bot.transcribe_file
        previous_cleanup = bot.clean_transcript
        bot.MANAGER_CHAT_ID = -1001234567890
        bot.transcribe_file = fake_transcribe
        bot.clean_transcript = fake_cleanup
        try:
            asyncio.run(bot.handle_private_message(update, context))
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id
            bot.transcribe_file = previous_transcribe
            bot.clean_transcript = previous_cleanup

        self.assertEqual(message.replies, [])
        self.assertEqual(len(message.copies), 1)
        self.assertIn("Распознавание...", message.copies[0]["caption"])
        self.assertEqual(len(manager_bot.edited_captions), 1)
        final_caption = manager_bot.edited_captions[0]["caption"]
        self.assertIn("Автор: Иван Иванов (@ivan, ID: 123)", final_caption)
        self.assertIn("Добрый день.", final_caption)
        self.assertEqual(manager_bot.sent_messages, [])

    def test_long_voice_transcript_continues_only_in_manager_group(self):
        manager_bot = FakeManagerBot()
        context = SimpleNamespace(bot=manager_bot)
        user = SimpleNamespace(id=123, full_name="Иван Иванов", username="ivan")
        long_text = "слово " * 400
        result = bot.TranscriptionResult(text=long_text.strip(), confidence=0.95)
        cleanup = bot.CleanupResult(text=long_text.strip())

        previous_manager_chat_id = bot.MANAGER_CHAT_ID
        bot.MANAGER_CHAT_ID = -1001234567890
        try:
            asyncio.run(
                bot.publish_audio_result(
                    context,
                    9001,
                    user,
                    result,
                    cleanup,
                    supports_caption=True,
                )
            )
        finally:
            bot.MANAGER_CHAT_ID = previous_manager_chat_id

        self.assertEqual(len(manager_bot.edited_captions), 1)
        caption = manager_bot.edited_captions[0]["caption"]
        self.assertLessEqual(bot.utf16_length(caption), 1024)
        self.assertIn("Продолжение транскрипции ниже", caption)
        self.assertGreaterEqual(len(manager_bot.sent_messages), 1)
        self.assertIn("Продолжение транскрипции", manager_bot.sent_messages[0]["text"])

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
