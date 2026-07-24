"""Run one media translation job using Azure Blob Storage for I/O."""

import argparse
import tempfile
from pathlib import Path

from translate_video import VideoTranslationPipeline
from src.storage import AzureBlobStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate one Azure Blob media object")
    parser.add_argument("--input-uri", required=True, help="az://container/path/input.mp4")
    parser.add_argument("--output-uri", required=True, help="az://container/path/output.mp4")
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-lang")
    parser.add_argument("--transcription", choices=["openai", "local"], default="openai")
    parser.add_argument("--translation-provider", choices=["deepseek", "openai"], default="deepseek")
    args = parser.parse_args()

    storage = AzureBlobStorage()
    with tempfile.TemporaryDirectory(prefix="videotranslator-") as temp_dir:
        work_dir = Path(temp_dir)
        suffix = Path(AzureBlobStorage.split_uri(args.input_uri)[1]).suffix or ".mp4"
        local_input = storage.download(args.input_uri, work_dir / f"input{suffix}")
        pipeline = VideoTranslationPipeline(
            target_language=args.target_lang,
            source_language=args.source_lang,
            output_dir=work_dir / "output",
            whisper_mode=args.transcription,
            translator_service=args.translation_provider,
            tts_mode="local",
        )
        result = pipeline.run(local_input)
        result_uri = storage.upload(Path(result["media_translated"]), args.output_uri)
        print(result_uri)


if __name__ == "__main__":
    main()
